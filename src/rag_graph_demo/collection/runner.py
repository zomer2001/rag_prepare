"""Pipeline orchestration: discover → policy gate → fetch → parse → normalize
→ dedupe/version → validate → clean+sidecar → run report.

The runner is the ONLY place stages are wired together, which keeps adapters
small and replaceable. It writes exclusively under data/collection/ and never
touches data/lightrag, qdrant_storage, or the document registry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .adapters import get_adapter
from .dedupe import RecordStore, SimHashIndex
from .http import DomainRateLimiter, HttpFetcher
from .models import (
    CandidateStatus,
    DiscoveryCandidate,
    RunError,
    RunManifest,
    RunMetrics,
    SourceConfig,
)
from .normalize import write_clean_and_sidecar
from .parsers import parse_by_mime
from .policy import PolicyGate, RobotsCache
from .quality import hard_validate, score_quality, write_review_entry
from .registry import SourceRegistry
from .scheduler import (
    RunLock,
    StateStore,
    append_discovery,
    ensure_collection_dirs,
    new_run_id,
    write_dead_letter,
    write_manifest,
)

CURSOR_SUCCESS_THRESHOLD = 0.9  # cursor advances only if ≥90% of items succeeded


@dataclass
class RunnerContext:
    live: bool
    limit: int | None
    collection_root: Path
    since: str | None

    # adapter-facing context protocol fields


class FixtureManifest:
    """Offline transport: maps (method, url) → fixture file.

    Fixture mode is the default; live network access requires explicit
    ``--live`` (brief §10: tests never touch the real network).
    """

    def __init__(self, fixtures_root: Path | str, source: SourceConfig) -> None:
        self.root = Path(fixtures_root) / (source.fixture_dir or source.source_id)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"fixture manifest missing: {manifest_path} "
                "(run with --live for real network access)"
            )
        self.mapping: dict[str, str] = json.loads(manifest_path.read_text(encoding="utf-8"))

    def resolve(self, method: str, url: str) -> Path | None:
        variants = {url, url.rstrip("/"), url + "/" if not url.endswith("/") else url}
        bases = {v.split("?", 1)[0] for v in variants}
        for candidate in list(variants) + list(bases):
            for key in (f"{method} {candidate}", candidate):
                if key in self.mapping:
                    return self.root / self.mapping[key]
        return None


def _fixture_client(manifest: FixtureManifest) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        path = manifest.resolve(request.method, str(request.url))
        if path is None or not path.exists():
            return httpx.Response(404, text=f"no fixture for {request.method} {request.url}")
        body = path.read_bytes()
        headers = {"Content-Type": _guess_content_type(path)}
        return httpx.Response(200, content=body, headers=headers, request=request)

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def _guess_content_type(path: Path) -> str:
    return {
        ".html": "text/html; charset=utf-8",
        ".json": "application/json",
        ".xml": "application/xml",
        ".pdf": "application/pdf",
        ".txt": "text/plain; charset=utf-8",
    }.get(path.suffix.lower(), "application/octet-stream")


class CollectionRunner:
    def __init__(
        self,
        registry: SourceRegistry,
        *,
        collection_root: Path | str = "data/collection",
        live: bool = False,
        fixtures_dir: Path | str = "tests/fixtures/collection",
        limit: int | None = None,
        since: str | None = None,
        resolve_dns: bool | None = None,
    ) -> None:
        self.registry = registry
        self.collection_root = ensure_collection_dirs(collection_root)
        self.live = live
        self.fixtures_dir = Path(fixtures_dir)
        self.limit = limit
        self.since = since
        self.resolve_dns = live if resolve_dns is None else resolve_dns
        self.state_store = StateStore(self.collection_root / "state")
        self.record_store = RecordStore(self.collection_root / "normalized")
        self.simhash = SimHashIndex(self.collection_root / "state" / "simhash_index.jsonl")
        self._clients: list[httpx.Client] = []

    # ------------------------------------------------------------------ API

    def discover_source(self, source_id: str) -> RunManifest:
        source = self.registry.get(source_id)
        manifest = self._new_manifest("discover", [source_id])
        state = self.state_store.load(source_id)
        fetcher = self._make_fetcher(source)
        adapter = get_adapter(source.adapter)
        ctx = self._ctx()
        try:
            candidates = adapter.discover(source, state, ctx, fetcher)
        except Exception as exc:
            manifest.errors.append(
                RunError(stage="discover", message=str(exc), retryable=True)
            )
            manifest.metrics.permanent_failures += 1
            self._finish(manifest, state)
            return manifest
        finally:
            self._close_clients()
        manifest.metrics.discovered = len(candidates)
        if state.health.get("policy_skip"):
            manifest.metrics.policy_skipped += 1
        append_discovery(
            self.collection_root / "discovery",
            source_id,
            [c.model_dump(mode="json") for c in candidates],
        )
        self._finish(manifest, state)
        return manifest

    def run_source(self, source_id: str, candidate_urls: list[str] | None = None) -> RunManifest:
        source = self.registry.get(source_id)
        manifest = self._new_manifest("run", [source_id])
        manifest.cursor_before = dict(self.state_store.load(source_id).cursor)
        try:
            with RunLock(self.collection_root / "state", source_id):
                self._run_locked(source, manifest, candidate_urls)
        except Exception as exc:
            manifest.errors.append(RunError(stage="run", message=str(exc), retryable=True))
            manifest.metrics.permanent_failures += 1
        finally:
            self._close_clients()
        manifest.cursor_after = dict(self.state_store.load(source_id).cursor)
        self._finish(manifest, self.state_store.load(source_id), advance_cursor=False)
        return manifest

    def validate_record(self, record_id: str) -> dict[str, Any]:
        latest = self.record_store.load_latest(record_id)
        if latest is None:
            return {"record_id": record_id, "valid": False, "errors": ["record not found"]}
        errors = hard_validate(latest)
        return {"record_id": record_id, "valid": not errors, "errors": errors}

    # ------------------------------------------------------------- internals

    def _run_locked(
        self,
        source: SourceConfig,
        manifest: RunManifest,
        candidate_urls: list[str] | None,
    ) -> None:
        metrics: RunMetrics = manifest.metrics
        state = self.state_store.load(source.source_id)
        adapter = get_adapter(source.adapter)
        fetcher = self._make_fetcher(source)
        gate = PolicyGate(
            robots=RobotsCache(fetcher=self._robots_fetcher(source)),
            resolve_dns=self.resolve_dns,
        )
        ctx = self._ctx()

        try:
            candidates = adapter.discover(source, state, ctx, fetcher)
        except Exception as exc:
            manifest.errors.append(RunError(stage="discover", message=str(exc), retryable=True))
            metrics.retryable_failures += 1
            self.state_store.save(state)
            return
        metrics.discovered = len(candidates)
        if state.health.get("policy_skip"):
            metrics.policy_skipped += 1
            self.state_store.save(state)
            return  # policy skip: nothing to do, recorded for the report
        append_discovery(
            self.collection_root / "discovery",
            source.source_id,
            [c.model_dump(mode="json") for c in candidates],
        )
        if candidate_urls:
            wanted = set(candidate_urls)
            candidates = [c for c in candidates if c.url in wanted]

        succeeded = 0
        attempted = 0
        for candidate in candidates:
            attempted += 1
            try:
                outcome = self._process_candidate(
                    source, adapter, gate, fetcher, state, candidate, manifest
                )
            except Exception as exc:  # per-item isolation: one bad item never kills the run
                metrics.permanent_failures += 1
                manifest.errors.append(
                    RunError(
                        stage="item",
                        message=str(exc),
                        url=candidate.url,
                        candidate_id=candidate.candidate_id,
                        retryable=False,
                    )
                )
                write_dead_letter(
                    self.collection_root / "dead_letter",
                    {
                        "source_id": source.source_id,
                        "candidate": candidate.model_dump(mode="json"),
                        "error": str(exc),
                        "run_id": manifest.run_id,
                    },
                )
                continue
            if outcome:
                succeeded += 1

        # Cursor advances only past the success threshold (brief §7.2).
        if attempted and succeeded / attempted >= CURSOR_SUCCESS_THRESHOLD:
            state.cursor["last_successful_run"] = manifest.started_at.isoformat()
            state.cursor["last_run_id"] = manifest.run_id
        state.health.update(
            {
                "last_run_id": manifest.run_id,
                "last_run_metrics": metrics.model_dump(),
                "last_error_count": len(manifest.errors),
            }
        )
        self.state_store.save(state)

    def _process_candidate(
        self,
        source: SourceConfig,
        adapter: Any,
        gate: PolicyGate,
        fetcher: HttpFetcher,
        state: Any,
        candidate: DiscoveryCandidate,
        manifest: RunManifest,
    ) -> bool:
        metrics = manifest.metrics
        candidate.status = CandidateStatus.ACCEPTED
        metrics.accepted += 1

        url = candidate.url
        if url is None:
            metrics.permanent_failures += 1
            manifest.errors.append(
                RunError(stage="policy", message="candidate has no fetchable URL",
                         candidate_id=candidate.candidate_id)
            )
            return False
        decision = gate.check(url, robots_url=source.policy.robots_url)
        if not decision.allowed:
            candidate.status = CandidateStatus.REJECTED
            candidate.policy_skip_reason = decision.reason
            metrics.policy_skipped += 1
            return False

        validators = state.validators.get(decision.canonical_url or url, {})
        fetch = fetcher.fetch(
            decision.canonical_url or url,
            etag=validators.get("etag"),
            last_modified=validators.get("last_modified"),
        )
        if fetch.not_modified:
            metrics.not_modified += 1
            return True
        if not fetch.ok:
            metrics.retryable_failures += 1 if fetch.retryable else 0
            metrics.permanent_failures += 0 if fetch.retryable else 1
            manifest.errors.append(
                RunError(
                    stage="fetch",
                    message=fetch.error or "fetch failed",
                    url=url,
                    candidate_id=candidate.candidate_id,
                    retryable=fetch.retryable,
                )
            )
            if not fetch.retryable:
                write_dead_letter(
                    self.collection_root / "dead_letter",
                    {
                        "source_id": source.source_id,
                        "candidate": candidate.model_dump(mode="json"),
                        "error": fetch.error,
                        "run_id": manifest.run_id,
                    },
                )
            return False

        metrics.fetched += 1
        metrics.bytes_downloaded += fetch.size_bytes
        if decision.canonical_url:
            state.validators[decision.canonical_url] = {
                k: v
                for k, v in {"etag": fetch.etag, "last_modified": fetch.last_modified}.items()
                if v
            }

        raw_bytes = Path(fetch.raw_path).read_bytes() if fetch.raw_path else b""
        parsed = parse_by_mime(
            fetch.mime_type or "application/octet-stream",
            raw_bytes=raw_bytes,
            raw_path=fetch.raw_path,
        )
        metrics.parsed += 1
        raw_payload = None
        if parsed.next_data and "payload" in parsed.next_data:
            raw_payload = parsed.next_data["payload"]

        record = adapter.build_record(source, candidate, fetch, parsed, raw_payload)
        record_dict = record.model_dump(mode="json")

        # near-duplicate linking (relations only; nothing deleted, §4.6)
        for near_id, _version, dist in self.simhash.candidates(
            record_dict["body_markdown"] or record_dict["title"],
            exclude_record_id=record_dict["record_id"],
        ):
            record_dict["relations"].append(
                {
                    "relation_type": "near_duplicate_candidate",
                    "target_record_id": near_id,
                    "machine_inferred": True,
                    "rationale": f"simhash hamming distance {dist}",
                }
            )
            metrics.near_duplicate_candidates += 1

        # hard validation never masked by quality scores (§4.7)
        errors = hard_validate(record_dict)
        quality = score_quality(record_dict)
        record_dict["quality"] = quality
        if errors:
            metrics.review_pending += 1
            write_review_entry(self.collection_root / "review_queue", record_dict, errors)
            manifest.errors.append(
                RunError(
                    stage="validate",
                    message="; ".join(errors),
                    url=url,
                    candidate_id=candidate.candidate_id,
                    retryable=False,
                )
            )
            return False

        is_new_record, is_new_version = self.record_store.save_version(record_dict)
        if not is_new_version:
            metrics.exact_duplicates += 1
            return True
        metrics.new_records += 1 if is_new_record else 0
        metrics.new_versions += 0 if is_new_record else 1

        clean_path, _sidecar = write_clean_and_sidecar(self.collection_root / "clean", record_dict)
        record_dict["content"]["clean_path"] = str(clean_path)
        self.record_store.update_latest(record_dict)  # same version_id, attach clean_path
        self.simhash.add(
            record_dict["record_id"],
            record_dict["version_id"],
            record_dict["body_markdown"] or record_dict["title"],
        )
        if quality["needs_review"]:
            metrics.review_pending += 1
            write_review_entry(
                self.collection_root / "review_queue",
                record_dict,
                ["quality: " + "; ".join(quality["warnings"] or ["low quality"])],
            )
        metrics.published_clean += 1
        return True

    # --------------------------------------------------------------- helpers

    def _ctx(self) -> RunnerContext:
        return RunnerContext(
            live=self.live,
            limit=self.limit,
            collection_root=self.collection_root,
            since=self.since,
        )

    def _new_manifest(self, mode: str, source_ids: list[str]) -> RunManifest:
        manifest = RunManifest(
            run_id=new_run_id(),
            mode=mode,  # type: ignore[arg-type]
            live=self.live,
            source_ids=source_ids,
            config_hash=self.registry.config_hash(),
            indexed=False,  # invariant for this phase
        )
        for sid in source_ids:
            try:
                src = self.registry.get(sid)
                manifest.adapter_versions[src.adapter] = src.adapter_version
            except Exception:
                continue
        return manifest

    def _make_fetcher(self, source: SourceConfig) -> HttpFetcher:
        policy = source.policy
        limiter = DomainRateLimiter(default_interval=policy.rate_limit_seconds)
        if self.live:
            client = httpx.Client(
                timeout=30.0,
                follow_redirects=True,
                max_redirects=10,
                headers={"User-Agent": "rag-prepare-osint-collector/0.1"},
            )
        else:
            client = _fixture_client(FixtureManifest(self.fixtures_dir, source))
        self._clients.append(client)
        return HttpFetcher(
            self.collection_root / "raw",
            client=client,
            rate_limiter=limiter,
            max_bytes=policy.max_bytes,
            adapter=source.adapter,
            adapter_version=source.adapter_version,
        )

    def _robots_fetcher(self, source: SourceConfig):
        if self.live:
            client = httpx.Client(timeout=15.0, follow_redirects=True)
            self._clients.append(client)

            def fetch(url: str) -> tuple[int, str]:
                response = client.get(url)
                return response.status_code, response.text

            return fetch

        manifest_root = self.fixtures_dir / (source.fixture_dir or source.source_id)

        def fetch(url: str) -> tuple[int, str]:
            path = manifest_root / "robots.txt"
            if path.exists():
                return 200, path.read_text(encoding="utf-8")
            return 404, ""

        return fetch

    def _close_clients(self) -> None:
        for client in self._clients:
            client.close()
        self._clients.clear()

    def _finish(self, manifest: RunManifest, state: Any, advance_cursor: bool = True) -> Path:
        from datetime import UTC, datetime

        manifest.finished_at = datetime.now(UTC)
        return write_manifest(self.collection_root / "runs", manifest)

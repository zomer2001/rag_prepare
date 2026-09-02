from __future__ import annotations

import json
import os
from typing import Any

import httpx
import streamlit as st

API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000/api/v1")
SHOW_EVOLINK = os.getenv("RAG_SHOW_EVOLINK", "true").strip().casefold() in {
    "1",
    "true",
    "yes",
    "on",
}
QUERY_MODES = {
    "综合检索（推荐）": "mix",
    "纯向量检索": "naive",
    "局部图谱检索": "local",
    "全局图谱分析": "global",
    "图谱混合检索": "hybrid",
}


def api_request(
    method: str,
    path: str,
    *,
    timeout: float = 30,
    **kwargs: Any,
) -> Any:
    try:
        with httpx.Client(base_url=API_BASE_URL, timeout=timeout, trust_env=False) as client:
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        st.error(f"API 请求失败：{detail}")
    except (httpx.HTTPError, ValueError) as exc:
        st.error(f"无法连接后端：{exc}")
    return None


def render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        st.info("本次回答没有返回引用。")
        return
    for citation in citations:
        page = citation.get("page_number")
        section = citation.get("section")
        location = f"第 {page} 页" if page else section or "文档级来源"
        title = citation.get("title") or citation.get("filename") or citation.get("file_path")
        with st.expander(f"[{citation.get('reference_id')}] {title} · {location}"):
            if citation.get("excerpt"):
                st.markdown(citation["excerpt"])
            st.caption(citation.get("file_path", ""))


def render_query_response(response: dict[str, Any]) -> None:
    if response.get("answer"):
        st.markdown(response["answer"])
    else:
        st.warning(response.get("message") or "没有生成回答。")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("模式", response.get("mode", "-"))
    col2.metric("耗时", f"{response.get('elapsed_ms', 0)} ms")
    col3.metric("实体", len(response.get("entities") or []))
    col4.metric("关系", len(response.get("relationships") or []))
    st.subheader("引用")
    render_citations(response.get("citations") or [])
    events = response.get("events") or []
    if events:
        st.subheader("事件通路")
        for event in events:
            st.markdown(
                f"- `{event.get('start_time') or '时间未知'}` "
                f"**{event.get('event_type')}** — {event.get('summary')} "
                f"（{event.get('status')} / {event.get('confidence', 0):.2f}）"
            )
        conflicts = response.get("event_conflicts") or []
        if conflicts:
            st.warning("事件索引发现来源立场冲突，请结合原文引用核验。")
    evidence_graph = response.get("evidence_graph") or {}
    if evidence_graph.get("nodes"):
        st.subheader("本次决策证据图")
        st.caption(
            "蓝色为实体，绿色为语义断言，橙色为事件；红色边表示事件—实体/断言的跨通路连接。"
        )
        st.graphviz_chart(graph_to_dot(evidence_graph), use_container_width=True)
    with st.expander("检索上下文与调试数据"):
        st.json(
            {
                "entities": response.get("entities") or [],
                "relationships": response.get("relationships") or [],
                "chunks": response.get("chunks") or [],
                "evidence_graph": evidence_graph,
                "metadata": response.get("metadata") or {},
            }
        )


def dot_escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def graph_to_dot(graph: dict[str, Any]) -> str:
    lines = [
        "digraph knowledge {",
        "rankdir=LR;",
        'node [shape=box, style="rounded,filled", fontname="Arial"];',
        'edge [fontname="Arial", color="#64748b"];',
    ]
    node_styles = {
        "entity": ('ellipse', "#dbeafe", "#2563eb"),
        "assertion": ('box', "#dcfce7", "#16a34a"),
        "event": ('hexagon', "#ffedd5", "#ea580c"),
    }
    for node in graph.get("nodes") or []:
        node_id = dot_escape(node.get("id", ""))
        labels = node.get("labels") or []
        label = node.get("label") or (labels[0] if labels else node_id)
        shape, fill, color = node_styles.get(
            str(node.get("type") or ""),
            ("box", "#f8fafc", "#64748b"),
        )
        lines.append(
            f'"{node_id}" [label="{dot_escape(label)}", shape="{shape}", '
            f'fillcolor="{fill}", color="{color}"];'
        )
    for edge in graph.get("edges") or []:
        source = dot_escape(edge.get("source", ""))
        target = dot_escape(edge.get("target", ""))
        edge_type = dot_escape(edge.get("type") or "related")
        color = "#dc2626" if edge.get("route") == "cross" else "#64748b"
        lines.append(
            f'"{source}" -> "{target}" [label="{edge_type}", color="{color}"];'
        )
    lines.append("}")
    return "\n".join(lines)


st.set_page_config(page_title="RAG Graph Demo", page_icon="🔎", layout="wide")
st.title("RAG Graph Demo")
st.caption("LightRAG + Qdrant + Docling + DeepSeek")

status_tab, documents_tab, query_tab, events_tab, compare_tab, graph_tab = st.tabs(
    ["系统状态", "文档管理", "知识库问答", "事件时间线", "模式对比", "知识图谱"]
)

with status_tab:
    st.subheader("运行状态")
    if st.button("刷新系统状态", type="primary"):
        st.session_state["diagnostics"] = api_request("GET", "/diagnostics", timeout=5)
    diagnostics = st.session_state.get("diagnostics") or api_request(
        "GET", "/diagnostics", timeout=5
    )
    if diagnostics:
        summary = diagnostics.get("summary") or {}
        if summary.get("ready"):
            st.success("基础服务已就绪")
        else:
            st.error("基础服务尚未就绪")
        deepseek = diagnostics.get("deepseek") or {}
        if deepseek.get("configured"):
            st.success(f"DeepSeek 已配置：{deepseek.get('chat_model')}")
        else:
            st.warning("DeepSeek API Key 尚未配置；可以上传解析，但不能完成 LightRAG 索引。")
        st.json(diagnostics)

with documents_tab:
    st.subheader("上传文档")
    files = st.file_uploader(
        "支持 Markdown、DOCX、文字型 PDF",
        type=["md", "markdown", "docx", "pdf"],
        accept_multiple_files=True,
    )
    auto_index = st.checkbox("上传后立即建立 LightRAG 索引", value=True)
    if st.button("上传", disabled=not files, type="primary"):
        for uploaded in files or []:
            result = api_request(
                "POST",
                f"/documents?auto_index={str(auto_index).lower()}",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=120,
            )
            if result:
                duplicate_text = "（重复文件，未再次索引）" if result.get("duplicate") else ""
                st.success(f"已接收 {uploaded.name}{duplicate_text}")

    col_refresh, col_jobs = st.columns(2)
    refresh = col_refresh.button("刷新文档状态")
    show_jobs = col_jobs.checkbox("显示任务", value=False)
    if refresh or "documents" not in st.session_state:
        st.session_state["documents"] = api_request("GET", "/documents", timeout=10) or []
    documents = st.session_state.get("documents", [])
    if documents:
        rows = [
            {
                "标题": item.get("title"),
                "文件": item.get("filename"),
                "状态": item.get("status"),
                "事件状态": item.get("event_index_status"),
                "事件数": item.get("event_count", 0),
                "页数": item.get("page_count"),
                "错误": item.get("error"),
                "ID": item.get("document_id"),
            }
            for item in documents
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        selected = st.selectbox(
            "文档操作",
            options=documents,
            format_func=lambda item: f"{item.get('title')} · {item.get('status')}",
        )
        col_preview, col_reindex, col_events, col_delete = st.columns(4)
        if col_preview.button("查看解析内容"):
            content = api_request(
                "GET", f"/documents/{selected['document_id']}/content", timeout=10
            )
            if content:
                st.markdown(content.get("content") or "尚未解析。")
        if col_reindex.button("重新索引"):
            result = api_request(
                "POST", f"/documents/{selected['document_id']}/reindex", timeout=10
            )
            if result:
                st.success(f"重建任务已创建：{result.get('job_id')}")
        if col_events.button("重建事件"):
            result = api_request(
                "POST",
                f"/documents/{selected['document_id']}/events/rebuild",
                timeout=10,
            )
            if result:
                st.success(f"事件重建任务已创建：{result.get('job_id')}")
        confirm_delete = col_delete.checkbox("确认删除")
        if col_delete.button("删除", disabled=not confirm_delete) and api_request(
            "DELETE", f"/documents/{selected['document_id']}", timeout=120
        ):
            st.success("文档已删除")
            st.session_state.pop("documents", None)
    else:
        st.info("尚未上传文档。")

    if show_jobs:
        jobs = api_request("GET", "/jobs", timeout=10) or []
        st.dataframe(jobs, use_container_width=True, hide_index=True)

with query_tab:
    st.subheader("知识库问答")
    question = st.text_area("问题", placeholder="结合这些报道，主要公司之间有哪些合作关系？")
    mode_label = st.selectbox("检索模式", list(QUERY_MODES))
    col_top_k, col_chunk_k, col_rerank = st.columns(3)
    top_k = col_top_k.slider("实体/关系召回", 1, 50, 20)
    chunk_top_k = col_chunk_k.slider("文本块召回", 1, 30, 10)
    enable_rerank = col_rerank.checkbox("启用 Reranker", value=False)
    use_events = st.checkbox("启用事件通路（时间筛选、事件链与冲突提示）", value=True)
    evidence_linking = False
    if SHOW_EVOLINK:
        evidence_linking = st.checkbox(
            "启用 ours_evolink（Neo4j跨通路证据图）",
            value=False,
            disabled=not use_events,
        )
    if st.button("开始问答", disabled=not question.strip(), type="primary"):
        with st.spinner("正在检索和生成答案…"):
            response = api_request(
                "POST",
                "/query",
                json={
                    "question": question,
                    "mode": QUERY_MODES[mode_label],
                    "top_k": top_k,
                    "chunk_top_k": chunk_top_k,
                    "enable_rerank": enable_rerank,
                    "use_events": use_events,
                    "evidence_linking": evidence_linking,
                },
                timeout=300,
            )
        if response:
            render_query_response(response)

with events_tab:
    st.subheader("事件时间线")
    st.caption("事件是带时间、参与者、状态和原文证据的关系断言；并非绝对事实。")
    col_entity, col_start, col_end = st.columns(3)
    event_entity = col_entity.text_input("实体/产品", placeholder="Anduril 或 Lattice")
    event_start = col_start.text_input("开始时间", placeholder="2023 或 2023-01")
    event_end = col_end.text_input("结束时间", placeholder="2026-12-31")
    event_question = st.text_input(
        "事件问题（可选）", placeholder="近三年 Anduril 产品经历了哪些关键变化？"
    )
    if st.button("查询事件时间线", type="primary"):
        timeline = api_request(
            "POST",
            "/events/query",
            json={
                "question": event_question,
                "entity": event_entity or None,
                "start_time": event_start or None,
                "end_time": event_end or None,
                "limit": 50,
                "sort_by": "time",
            },
            timeout=30,
        )
        if timeline:
            events = timeline.get("events") or []
            if not events:
                st.info("没有匹配事件。请先完成文档事件索引，或放宽筛选条件。")
            for event in events:
                with st.container(border=True):
                    st.markdown(
                        f"**{event.get('start_time') or '时间未知'} · "
                        f"{event.get('event_type')}**"
                    )
                    st.write(event.get("summary"))
                    st.caption(
                        f"主体：{', '.join(event.get('actors') or []) or '-'} ｜ "
                        f"对象：{', '.join(event.get('objects') or []) or '-'} ｜ "
                        f"状态：{event.get('status')} ｜ "
                        f"置信度：{event.get('confidence', 0):.2f}"
                    )
            with st.expander("事件关系、证据与冲突"):
                st.json(
                    {
                        "relations": timeline.get("relations") or [],
                        "mentions": timeline.get("mentions") or [],
                        "conflicts": timeline.get("conflicts") or [],
                    }
                )

with compare_tab:
    st.subheader("同一问题横向比较")
    compare_question = st.text_area(
        "比较问题",
        placeholder="这些报道反映了 AI 行业的哪些趋势？",
        key="compare_question",
    )
    selected_labels = st.multiselect(
        "比较模式",
        list(QUERY_MODES),
        default=["纯向量检索", "局部图谱检索", "全局图谱分析", "综合检索（推荐）"],
    )
    if st.button("运行比较", disabled=not compare_question.strip() or not selected_labels):
        with st.spinner("正在顺序运行多个检索模式…"):
            responses = api_request(
                "POST",
                "/query/compare",
                json={
                    "question": compare_question,
                    "modes": [QUERY_MODES[label] for label in selected_labels],
                },
                timeout=600,
            )
        if responses:
            tabs = st.tabs([item.get("mode", "unknown") for item in responses])
            for tab, response in zip(tabs, responses, strict=True):
                with tab:
                    render_query_response(response)

with graph_tab:
    st.subheader("知识图谱")
    col_label, col_depth, col_nodes = st.columns(3)
    label = col_label.text_input("起始实体，`*` 表示全图", value="*")
    max_depth = col_depth.slider("最大深度", 0, 5, 2)
    max_nodes = col_nodes.slider("最大节点数", 10, 300, 100, step=10)
    if st.button("加载图谱", type="primary"):
        graph = api_request(
            "GET",
            "/graph",
            params={"label": label, "max_depth": max_depth, "max_nodes": max_nodes},
            timeout=120,
        )
        if graph:
            st.caption(
                f"节点 {len(graph.get('nodes') or [])} · 关系 {len(graph.get('edges') or [])}"
                f" · 截断：{graph.get('is_truncated', False)}"
            )
            if graph.get("nodes"):
                st.graphviz_chart(graph_to_dot(graph), use_container_width=True)
            with st.expander("图谱原始数据"):
                st.code(json.dumps(graph, ensure_ascii=False, indent=2), language="json")

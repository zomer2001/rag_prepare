"""Collection adapters. Each adapter documents its verified entry points,
pagination, rate limits and fallbacks in docs/source_playbook.md."""

from __future__ import annotations

from .anduril_site import AndurilSiteAdapter
from .generic_web import GenericWebAdapter
from .patents import PatentsViewAdapter
from .rss_feed import RssFeedAdapter
from .scholarly import OpenAlexAdapter
from .sitemap_archive import SitemapArchiveAdapter
from .usaspending import USASpendingAdapter

ADAPTERS = {
    "anduril_site": AndurilSiteAdapter,
    "usaspending": USASpendingAdapter,
    "openalex": OpenAlexAdapter,
    "patentsview": PatentsViewAdapter,
    "rss_feed": RssFeedAdapter,
    "sitemap_archive": SitemapArchiveAdapter,
    "generic_web": GenericWebAdapter,
}


def get_adapter(name: str):
    try:
        return ADAPTERS[name]()
    except KeyError as exc:
        raise KeyError(f"unknown adapter {name!r}; available: {sorted(ADAPTERS)}") from exc

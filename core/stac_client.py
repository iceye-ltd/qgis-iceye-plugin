"""STAC catalog client for ICEYE open data."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

DEFAULT_CATALOG_URL = (
    "https://iceye-open-data-catalog.s3-us-west-2.amazonaws.com/catalog.json"
)

_CACHE: dict[str, Any] = {
    "catalog_url": None,
    "collections": None,
}
_ITEM_CACHE: dict[str, dict[str, Any]] = {}


def fetch_catalog(
    catalog_url: str, force_refresh: bool = False
) -> list[dict[str, Any]]:
    """Fetch STAC catalog and return list of collections with items.

    Parameters
    ----------
    catalog_url : str
        URL of the STAC catalog JSON.
    force_refresh : bool, optional
        If True, bypass cache and fetch fresh data. Default is False.

    Returns
    -------
    list of dict
        List of collection dicts with id, title, href, and items.
    """
    if (
        not force_refresh
        and _CACHE["catalog_url"] == catalog_url
        and _CACHE["collections"] is not None
    ):
        return _CACHE["collections"]

    catalog = _fetch_json(catalog_url)
    collection_links = _filter_links(catalog, rels=("child", "collection"))
    collections = []
    for link in collection_links:
        collection_url = _resolve_href(catalog_url, link["href"])
        collection = _fetch_json(collection_url)
        items = _extract_items(collection, collection_url)
        collections.append(
            {
                "id": collection.get("id"),
                "title": collection.get("title"),
                "href": collection_url,
                "items": items,
            }
        )

    _CACHE["catalog_url"] = catalog_url
    _CACHE["collections"] = collections
    return collections


def fetch_item(item_url: str, force_refresh: bool = False) -> dict[str, Any]:
    """Fetch a single STAC item by URL.

    Parameters
    ----------
    item_url : str
        URL of the STAC item JSON.
    force_refresh : bool, optional
        If True, bypass cache. Default is False.

    Returns
    -------
    dict
        STAC item as dict.
    """
    if not force_refresh and item_url in _ITEM_CACHE:
        return _ITEM_CACHE[item_url]

    item = _fetch_json(item_url)
    _ITEM_CACHE[item_url] = item
    return item


def _fetch_json(url: str) -> dict[str, Any]:
    """Fetch JSON from URL and parse as dict."""
    request = Request(url, headers={"User-Agent": "ICEYE-QGIS-Plugin"})
    with urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
        return json.loads(payload)


def _filter_links(
    obj: dict[str, Any], rels: tuple[str, ...] | None = None
) -> list[dict[str, Any]]:
    """Filter links in a STAC object by rel types."""
    rel_set = set(rels or [])
    links = obj.get("links", [])
    if not rel_set:
        return links
    return [link for link in links if link.get("rel") in rel_set]


def _extract_items(collection: dict[str, Any], base_url: str) -> list[dict[str, str]]:
    """Extract item links from a STAC collection as list of {id, href} dicts."""
    items: list[dict[str, str]] = []
    for link in _filter_links(collection, rels=("item",)):
        href = link.get("href")
        if not href:
            continue
        item_url = _resolve_href(base_url, href)
        item_id = link.get("title") or link.get("id") or _id_from_href(item_url)
        items.append({"id": item_id, "href": item_url})
    return items


def _resolve_href(base_url: str, href: str) -> str:
    """Resolve relative href against base URL."""
    return urljoin(base_url, href)


def _id_from_href(href: str) -> str:
    """Extract item ID from href path (filename without .json)."""
    parsed = urlparse(href)
    name = parsed.path.split("/")[-1]
    if name.endswith(".json"):
        name = name[:-5]
    return name or href

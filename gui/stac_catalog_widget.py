"""STAC catalog dock widget for browsing and loading ICEYE items."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsJsonUtils,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsSymbol,
    QgsTask,
    QgsVectorLayer,
)
from qgis.PyQt import uic
from qgis.PyQt.QtCore import QByteArray, QStandardPaths, Qt, QVariant
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QAbstractItemView, QTreeWidgetItem

from ..core import stac_client

WIDGET, BASE = uic.loadUiType(
    str(Path(__file__).resolve().parent.parent / "ui" / "stac_catalog_widget.ui")
)


class StacCatalogTask(QgsTask):
    """Background task to fetch STAC catalog."""

    def __init__(self, widget, catalog_url: str, force_refresh: bool) -> None:
        super().__init__("Load STAC catalog", QgsTask.CanCancel)
        self.widget = widget
        self.catalog_url = catalog_url
        self.force_refresh = force_refresh
        self.collections = None
        self.error = None

    def run(self) -> bool:
        """Fetch catalog from STAC URL."""
        try:
            self.collections = stac_client.fetch_catalog(
                self.catalog_url, force_refresh=self.force_refresh
            )
            return True
        except Exception as exc:
            self.error = str(exc)
            return False

    def finished(self, success: bool) -> None:
        """Notify widget when catalog load completes."""
        if self.widget:
            self.widget.on_catalog_loaded(success, self.collections, self.error)


class StacItemsGeoJsonTask(QgsTask):
    """Background task to fetch STAC items by href."""

    def __init__(self, widget, item_hrefs: list[str]) -> None:
        super().__init__("Load STAC items", QgsTask.CanCancel)
        self.widget = widget
        self.item_hrefs = item_hrefs
        self.items = []
        self.error = None

    def run(self) -> bool:
        """Fetch STAC items by href."""
        try:
            for href in self.item_hrefs:
                if self.isCanceled():
                    return False
                item = stac_client.fetch_item(href)
                self.items.append(item)
            return True
        except Exception as exc:
            self.error = str(exc)
            return False

    def finished(self, success: bool) -> None:
        """Notify widget when items load completes."""
        if self.widget:
            self.widget.on_items_loaded(success, self.items, self.error)


def _slc_filename_from_url(url: str, item_id: str | None) -> str:
    """Extract filename from SLC URL for download."""
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        name = (item_id or "slc") + ".tif"
    if "." not in name:
        name = name + ".tif"
    return name


class StacSlcSingleDownloadTask(QgsTask):
    """Background task to download a single SLC file."""

    def __init__(
        self,
        url: str,
        path: str,
        item_id: str,
        parent_results: list,
    ) -> None:
        super().__init__(f"Download SLC {item_id}", QgsTask.CanCancel)
        self.url = url
        self.path = path
        self.item_id = item_id
        self.parent_results = parent_results
        self.result: dict | None = None

    def run(self) -> bool:
        """Download SLC file to path."""
        dest = Path(self.path)
        if dest.exists():
            self.result = {"id": self.item_id, "path": self.path, "error": None}
            self.setProgress(100)
            return True
        try:
            if self._download():
                self.result = {"id": self.item_id, "path": self.path, "error": None}
                return True
            return False
        except Exception as exc:
            self.result = {
                "id": self.item_id,
                "path": None,
                "error": str(exc),
            }
            return False

    def finished(self, success: bool) -> None:
        """Append result to parent's shared list."""
        if self.result is not None:
            self.parent_results.append(self.result)

    def _download(self) -> bool:
        """Download file from URL."""
        request = Request(self.url, headers={"User-Agent": "ICEYE-QGIS-Plugin"})
        with (
            urlopen(request, timeout=60) as response,
            Path(self.path).open("wb") as out,
        ):
            total_length = response.getheader("Content-Length")
            total_bytes = int(total_length) if total_length else None
            chunk_size = 256 * 1024
            downloaded = 0
            while True:
                if self.isCanceled():
                    return False
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                if total_bytes:
                    downloaded += len(chunk)
                    fraction = min(downloaded / total_bytes, 1.0)
                    self.setProgress(int(fraction * 100))
            if self.isCanceled():
                dest_path = Path(self.path)
                if dest_path.exists():
                    dest_path.unlink()
                return False
            self.setProgress(100)
            return True


class StacSlcBatchTask(QgsTask):
    """Parent task that coordinates multiple SLC download subtasks."""

    def __init__(
        self,
        widget,
        download_items: list[tuple[str, str, str]],
        error_results: list[dict],
    ) -> None:
        super().__init__("Load SLC assets", QgsTask.CanCancel)
        self.widget = widget
        self.download_items = download_items
        self.error_results = error_results
        self.results: list[dict] = []

        for url, path, item_id in download_items:
            subtask = StacSlcSingleDownloadTask(
                url=url,
                path=path,
                item_id=item_id,
                parent_results=self.results,
            )
            self.addSubTask(subtask, [], QgsTask.ParentDependsOnSubTask)

    def run(self) -> bool:
        """No work here – all done in subtasks."""
        return True

    def finished(self, success: bool) -> None:
        """Notify widget when all SLC downloads complete."""
        if self.widget:
            all_results = self.error_results + self.results
            self.widget.on_slc_loaded(success, all_results, None)


class StacItemsQlkTask(QgsTask):
    """Background task to download QLK (Quick Look) assets from STAC items."""

    def __init__(self, widget, items: list[dict], cache_dir: str) -> None:
        super().__init__("Load QLK assets", QgsTask.CanCancel)
        self.widget = widget
        self.items = items
        self.cache_dir = cache_dir
        self.results = []
        self.error = None

    def run(self) -> bool:
        """Download QLK assets from STAC items to cache directory."""
        try:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            total = len(self.items) or 1
            for index, item in enumerate(self.items):
                if self.isCanceled():
                    return False
                asset = self.widget._qlk_asset_from_item(item)
                if not asset:
                    self.results.append(
                        {
                            "id": item.get("id", "unknown"),
                            "path": None,
                            "error": "No QLK asset found.",
                        }
                    )
                    continue
                qlk_url = asset.get("href")
                if not qlk_url:
                    self.results.append(
                        {
                            "id": item.get("id", "unknown"),
                            "path": None,
                            "error": "QLK asset has no href.",
                        }
                    )
                    continue
                filename = self._filename_from_url(
                    qlk_url, item.get("id"), default_ext="tif"
                )
                local_path = Path(self.cache_dir) / filename
                if not local_path.exists():
                    if not self._download_file_with_progress(
                        qlk_url, str(local_path), index, total
                    ):
                        return False
                else:
                    self.setProgress(((index + 1) / total) * 100)
                self.results.append(
                    {
                        "id": item.get("id", "unknown"),
                        "path": str(local_path),
                        "error": None,
                    }
                )
            return True
        except Exception as exc:
            self.error = str(exc)
            return False

    def finished(self, success: bool) -> None:
        """Notify widget when QLK download completes."""
        if self.widget:
            self.widget.on_qlk_loaded(success, self.results, self.error)

    def _filename_from_url(
        self, url: str, item_id: str | None, default_ext: str = "tif"
    ) -> str:
        parsed = urlparse(url)
        name = Path(parsed.path).name
        if not name:
            name = (item_id or "qlk") + f".{default_ext}"
        if "." not in name:
            name = name + f".{default_ext}"
        return name

    def _download_file_with_progress(
        self, url: str, destination: str, item_index: int, total_items: int
    ) -> bool:
        request = Request(url, headers={"User-Agent": "ICEYE-QGIS-Plugin"})
        try:
            with (
                urlopen(request, timeout=60) as response,
                Path(destination).open("wb") as out,
            ):
                total_length = response.getheader("Content-Length")
                total_bytes = int(total_length) if total_length else None
                chunk_size = 256 * 1024
                downloaded = 0
                base_start = (item_index / total_items) * 100
                base_end = ((item_index + 1) / total_items) * 100
                while True:
                    if self.isCanceled():
                        break
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    if total_bytes:
                        downloaded += len(chunk)
                        fraction = min(downloaded / total_bytes, 1.0)
                        self.setProgress(
                            base_start + (base_end - base_start) * fraction
                        )
                if self.isCanceled():
                    dest_path = Path(destination)
                    if dest_path.exists():
                        dest_path.unlink()
                    return False
                self.setProgress(base_end)
                return True
        except Exception:
            dest_path = Path(destination)
            if dest_path.exists():
                dest_path.unlink()
            raise


class StacCatalogWidget(BASE, WIDGET):
    """Dock widget for browsing a STAC catalog."""

    def __init__(
        self,
        parent=None,
        catalog_url: str | None = None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self.catalog_url = catalog_url or stac_client.DEFAULT_CATALOG_URL
        self._active_task = None
        self._active_items_task = None
        self._active_slc_task = None
        self._active_qlk_task = None

        self.refreshButton.clicked.connect(self.refresh_catalog)
        self.loadGeojsonButton.clicked.connect(self.load_selected_geojson)
        self.loadSlcButton.clicked.connect(self.load_selected_slc)
        self.filterLineEdit.textChanged.connect(self.apply_filter)

        self.catalogTree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.refresh_catalog()

    def refresh_catalog(self, force_refresh: bool = False) -> None:
        """Load or refresh the STAC catalog tree."""
        if self._active_task and self._active_task.isActive():
            return

        self.statusLabel.setText("Loading STAC catalog...")
        self.refreshButton.setEnabled(False)

        task = StacCatalogTask(self, self.catalog_url, force_refresh)
        self._active_task = task
        QgsApplication.taskManager().addTask(task)

    def load_selected_geojson(self) -> None:
        """Load selected items as QLK raster (when present) or footprint layers."""
        if (self._active_items_task and self._active_items_task.isActive()) or (
            self._active_qlk_task and self._active_qlk_task.isActive()
        ):
            return

        item_hrefs = []
        for item in self.catalogTree.selectedItems():
            if item.parent() is None:
                continue
            href = item.data(0, Qt.UserRole)
            if href:
                item_hrefs.append(href)

        if not item_hrefs:
            self.statusLabel.setText("Select one or more items to load.")
            return

        self.statusLabel.setText("Loading selected preview...")
        self.loadGeojsonButton.setEnabled(False)

        task = StacItemsGeoJsonTask(self, item_hrefs)
        self._active_items_task = task
        QgsApplication.taskManager().addTask(task)

    def load_selected_slc(self) -> None:
        """Download and load selected items as SLC raster layers."""
        if self._active_slc_task and self._active_slc_task.isActive():
            return

        item_hrefs = []
        for item in self.catalogTree.selectedItems():
            if item.parent() is None:
                continue
            href = item.data(0, Qt.UserRole)
            if href:
                item_hrefs.append(href)

        if not item_hrefs:
            self.statusLabel.setText("Select one or more items to load SLC.")
            return

        cache_dir = self._slc_cache_dir()
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        download_items: list[tuple[str, str, str]] = []
        error_results: list[dict] = []

        for href in item_hrefs:
            try:
                stac_item = stac_client.fetch_item(href)
                asset = self._slc_asset_from_item(stac_item)
                if not asset:
                    error_results.append(
                        {
                            "id": stac_item.get("id") or href,
                            "path": None,
                            "error": "No SLC asset found.",
                        }
                    )
                    continue
                slc_url = asset.get("href")
                if not slc_url:
                    error_results.append(
                        {
                            "id": stac_item.get("id") or href,
                            "path": None,
                            "error": "SLC asset has no href.",
                        }
                    )
                    continue
                item_id = stac_item.get("id") or href
                filename = _slc_filename_from_url(slc_url, item_id)
                local_path = str(cache_dir / filename)
                download_items.append((slc_url, local_path, item_id))
            except Exception as exc:
                error_results.append(
                    {
                        "id": href,
                        "path": None,
                        "error": str(exc),
                    }
                )

        if not download_items and not error_results:
            self.statusLabel.setText("Select one or more items to load SLC.")
            return

        self.statusLabel.setText(f"Loading SLC to {cache_dir} ...")
        self.loadSlcButton.setEnabled(False)

        task = StacSlcBatchTask(
            self,
            download_items=download_items,
            error_results=error_results,
        )
        self._active_slc_task = task
        QgsApplication.taskManager().addTask(task)

    def on_catalog_loaded(self, success: bool, collections, error: str | None) -> None:
        """Handle catalog load completion."""
        self.refreshButton.setEnabled(True)
        self._active_task = None

        if not success:
            message = error or "Failed to load STAC catalog."
            self.statusLabel.setText(message)
            QgsMessageLog.logMessage(message, "ICEYE Toolbox", level=Qgis.Warning)
            return

        self.catalogTree.clear()
        for collection in collections:
            collection_label = collection.get("title") or collection.get("id") or "N/A"
            collection_item = QTreeWidgetItem([collection_label])
            collection_item.setData(0, Qt.UserRole, collection.get("href"))
            self.catalogTree.addTopLevelItem(collection_item)

            for item in collection.get("items", []):
                item_label = item.get("id") or "N/A"
                item_node = QTreeWidgetItem([item_label])
                item_node.setData(0, Qt.UserRole, item.get("href"))
                collection_item.addChild(item_node)

        self.catalogTree.expandAll()
        self.apply_filter(self.filterLineEdit.text())
        self.statusLabel.setText("Loaded STAC catalog.")

    def on_items_loaded(self, success: bool, items, error: str | None) -> None:
        """Handle items load completion. Load QLK when present, else footprint."""
        self._active_items_task = None

        if not success:
            self.loadGeojsonButton.setEnabled(True)
            message = error or "Failed to load selected items."
            self.statusLabel.setText(message)
            QgsMessageLog.logMessage(message, "ICEYE Toolbox", level=Qgis.Warning)
            return

        qlk_items = []
        footprint_items = []
        for item in items:
            if self._qlk_asset_from_item(item):
                qlk_items.append(item)
            else:
                footprint_items.append(item)

        loaded = 0
        for item in footprint_items:
            if self._add_item_layer(item):
                loaded += 1

        if qlk_items:
            self.statusLabel.setText(
                f"Loading {len(qlk_items)} QLK to {self._qlk_cache_dir()} ..."
            )
            task = StacItemsQlkTask(self, qlk_items, str(self._qlk_cache_dir()))
            self._active_qlk_task = task
            QgsApplication.taskManager().addTask(task)
        else:
            self.loadGeojsonButton.setEnabled(True)
            self.statusLabel.setText(f"Loaded {loaded} preview item(s).")

    def on_slc_loaded(self, success: bool, results, error: str | None) -> None:
        """Handle SLC download completion."""
        self.loadSlcButton.setEnabled(True)
        self._active_slc_task = None

        if not success:
            message = error or "Failed to load SLC assets."
            self.statusLabel.setText(message)
            QgsMessageLog.logMessage(message, "ICEYE Toolbox", level=Qgis.Warning)
            return

        loaded = 0
        for result in results:
            item_id = result.get("id")
            error_text = result.get("error")
            path = result.get("path")
            if error_text:
                QgsMessageLog.logMessage(
                    f"SLC load failed for {item_id}: {error_text}",
                    "ICEYE Toolbox",
                    level=Qgis.Warning,
                )
                continue
            if not path:
                continue
            layer = QgsRasterLayer(path, item_id or "SLC")
            if not layer.isValid():
                QgsMessageLog.logMessage(
                    f"SLC layer invalid for {item_id}.",
                    "ICEYE Toolbox",
                    level=Qgis.Warning,
                )
                continue
            QgsProject.instance().addMapLayer(layer)
            loaded += 1

        self.statusLabel.setText(f"Loaded {loaded} SLC item(s).")

    def on_qlk_loaded(self, success: bool, results, error: str | None) -> None:
        """Handle QLK download completion."""
        self.loadGeojsonButton.setEnabled(True)
        self._active_qlk_task = None

        if not success:
            message = error or "Failed to load QLK assets."
            self.statusLabel.setText(message)
            QgsMessageLog.logMessage(message, "ICEYE Toolbox", level=Qgis.Warning)
            return

        loaded = 0
        for result in results:
            item_id = result.get("id")
            error_text = result.get("error")
            path = result.get("path")
            if error_text:
                QgsMessageLog.logMessage(
                    f"QLK load failed for {item_id}: {error_text}",
                    "ICEYE Toolbox",
                    level=Qgis.Warning,
                )
                continue
            if not path:
                continue
            layer_name = f"{item_id} QLK" if item_id else "QLK"
            layer = QgsRasterLayer(path, layer_name)
            if not layer.isValid():
                QgsMessageLog.logMessage(
                    f"QLK layer invalid for {item_id}.",
                    "ICEYE Toolbox",
                    level=Qgis.Warning,
                )
                continue
            QgsProject.instance().addMapLayer(layer)
            loaded += 1

        self.statusLabel.setText(f"Loaded {loaded} QLK item(s).")

    def apply_filter(self, text: str) -> None:
        """Filter catalog tree by search text."""
        filter_text = (text or "").strip().lower()
        for index in range(self.catalogTree.topLevelItemCount()):
            item = self.catalogTree.topLevelItem(index)
            self._filter_item(item, filter_text)

    def _filter_item(self, item: QTreeWidgetItem, filter_text: str) -> bool:
        if not filter_text:
            item.setHidden(False)
            for i in range(item.childCount()):
                self._filter_item(item.child(i), filter_text)
            return True

        item_match = filter_text in item.text(0).lower()
        child_match = False
        for i in range(item.childCount()):
            child_visible = self._filter_item(item.child(i), filter_text)
            child_match = child_match or child_visible

        visible = item_match or child_match
        item.setHidden(not visible)
        return visible

    def _add_item_layer(self, item: dict) -> bool:
        geometry = item.get("geometry")
        if geometry is None and item.get("bbox"):
            geometry = self._geometry_from_bbox(item.get("bbox"))
        if geometry is None:
            QgsMessageLog.logMessage(
                f"Item {item.get('id')} has no geometry.",
                "ICEYE Toolbox",
                level=Qgis.Warning,
            )
            return False

        geom_type = self._geometry_type(geometry)
        layer_name = item.get("id") or "STAC Item"
        layer = QgsVectorLayer(f"{geom_type}?crs=EPSG:4326", layer_name, "memory")
        if not layer.isValid():
            QgsMessageLog.logMessage(
                f"Failed to create layer for {layer_name}.",
                "ICEYE Toolbox",
                level=Qgis.Warning,
            )
            return False

        self._apply_layer_style(layer)

        fields = QgsFields()
        fields.append(QgsField("id", QVariant.String))
        fields.append(QgsField("collection", QVariant.String))
        fields.append(QgsField("datetime", QVariant.String))
        layer.dataProvider().addAttributes(fields)
        layer.updateFields()

        feature = QgsFeature(fields)
        feature.setAttribute("id", item.get("id"))
        feature.setAttribute("collection", item.get("collection"))
        feature.setAttribute("datetime", item.get("properties", {}).get("datetime"))
        qgs_geometry = self._geometry_from_geojson(geometry)
        if (qgs_geometry is None or qgs_geometry.isEmpty()) and item.get("bbox"):
            qgs_geometry = self._geometry_from_bbox_qgs(item.get("bbox"))
        if qgs_geometry is None or qgs_geometry.isEmpty():
            QgsMessageLog.logMessage(
                f"Failed to parse geometry for {layer_name}.",
                "ICEYE Toolbox",
                level=Qgis.Warning,
            )
            return False
        feature.setGeometry(qgs_geometry)
        layer.dataProvider().addFeature(feature)
        layer.updateExtents()

        QgsProject.instance().addMapLayer(layer)
        return True

    def _geometry_type(self, geometry: dict) -> str:
        geom_type = (geometry.get("type") or "Unknown").lower()
        mapping = {
            "point": "Point",
            "multipoint": "MultiPoint",
            "linestring": "LineString",
            "multilinestring": "MultiLineString",
            "polygon": "Polygon",
            "multipolygon": "MultiPolygon",
            "geometrycollection": "GeometryCollection",
        }
        return mapping.get(geom_type, "GeometryCollection")

    def _geometry_from_geojson(self, geometry: dict) -> QgsGeometry | None:
        geojson = json.dumps(geometry)
        if hasattr(QgsGeometry, "fromGeoJson"):
            return QgsGeometry.fromGeoJson(geojson)
        if hasattr(QgsJsonUtils, "geometryFromGeoJson"):
            return QgsJsonUtils.geometryFromGeoJson(geojson)
        if hasattr(QgsGeometry, "fromJson"):
            return QgsGeometry.fromJson(QByteArray(geojson.encode("utf-8")))
        return None

    def _geometry_from_bbox(self, bbox: list) -> dict | None:
        if not bbox or len(bbox) < 4:
            return None
        minx, miny, maxx, maxy = bbox[:4]
        ring = [
            [minx, miny],
            [minx, maxy],
            [maxx, maxy],
            [maxx, miny],
            [minx, miny],
        ]
        return {"type": "Polygon", "coordinates": [ring]}

    def _geometry_from_bbox_qgs(self, bbox: list) -> QgsGeometry | None:
        if not bbox or len(bbox) < 4:
            return None
        minx, miny, maxx, maxy = bbox[:4]
        return QgsGeometry.fromRect(QgsRectangle(minx, miny, maxx, maxy))

    def _apply_layer_style(self, layer: QgsVectorLayer) -> None:
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        if symbol is None:
            return
        teal = QColor(0, 128, 128, 100)
        symbol.setColor(teal)
        if hasattr(symbol, "setOpacity"):
            symbol.setOpacity(0.6)
        if hasattr(symbol, "setStrokeColor"):
            symbol.setStrokeColor(QColor(0, 96, 96, 180))
        if hasattr(symbol, "setStrokeWidth"):
            symbol.setStrokeWidth(0.6)
        layer.renderer().setSymbol(symbol)

    def _slc_cache_dir(self) -> Path:
        base = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
        if not base:
            base = Path("~/.local/share/QGIS").expanduser()
        return Path(base) / "iceye_toolbox" / "slc_cache"

    def _qlk_cache_dir(self) -> Path:
        base = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
        if not base:
            base = Path("~/.local/share/QGIS").expanduser()
        return Path(base) / "iceye_toolbox" / "qlk_cache"

    def _slc_asset_from_item(self, item: dict) -> dict | None:
        assets = item.get("assets") or {}
        for key in ("slc-cog", "slc"):
            asset = assets.get(key)
            if asset:
                return asset
        for asset in assets.values():
            title = (asset.get("title") or "").lower()
            asset_type = (asset.get("type") or "").lower()
            if "slc" in title and ("tiff" in asset_type or "geotiff" in asset_type):
                return asset
        return None

    def _qlk_asset_from_item(self, item: dict) -> dict | None:
        assets = item.get("assets") or {}
        for key in ("qlk-cog", "qlk"):
            asset = assets.get(key)
            if asset:
                return asset
        for asset in assets.values():
            title = (asset.get("title") or "").lower()
            asset_type = (asset.get("type") or "").lower()
            if "quicklook" in title or "qlk" in title:
                if "tiff" in asset_type or "geotiff" in asset_type:
                    return asset
        return None

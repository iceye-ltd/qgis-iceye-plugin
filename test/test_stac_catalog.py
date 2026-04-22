# coding=utf-8
"""Tests for STAC catalog widget, including Load Preview / QLK behavior."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QTreeWidgetItem

from iceye_toolbox.gui.stac_catalog_widget import (
    StacCatalogWidget,
    StacItemsQlkTask,
    StacSlcBatchTask,
    StacSlcSingleDownloadTask,
    _slc_filename_from_url,
)


@pytest.fixture
def stac_widget(qgis_iface):
    """Create StacCatalogWidget with catalog refresh mocked to avoid network."""
    with patch.object(StacCatalogWidget, "refresh_catalog"):
        widget = StacCatalogWidget(parent=None)
    return widget


class TestQlkAssetFromItem:
    """Tests for _qlk_asset_from_item helper."""

    def test_returns_asset_when_qlk_cog_present(self, stac_widget):
        """Should return qlk-cog asset when present."""
        item = {
            "id": "ICEYE_TEST_123",
            "assets": {
                "qlk-cog": {
                    "href": "https://example.com/item_QLK.tif",
                    "type": "application/x-geotiff",
                    "title": "Quicklook COG",
                },
            },
        }
        asset = stac_widget._qlk_asset_from_item(item)
        assert asset is not None
        assert asset["href"] == "https://example.com/item_QLK.tif"
        assert asset["title"] == "Quicklook COG"

    def test_returns_asset_when_qlk_key_present(self, stac_widget):
        """Should return qlk asset when qlk key present (no -cog suffix)."""
        item = {
            "id": "ICEYE_TEST_456",
            "assets": {
                "qlk": {
                    "href": "https://example.com/item_qlk.tif",
                    "type": "application/x-geotiff",
                },
            },
        }
        asset = stac_widget._qlk_asset_from_item(item)
        assert asset is not None
        assert asset["href"] == "https://example.com/item_qlk.tif"

    def test_returns_asset_by_title_fallback(self, stac_widget):
        """Should find QLK asset by title when keys not matched."""
        item = {
            "id": "ICEYE_TEST_789",
            "assets": {
                "custom-qlk": {
                    "href": "https://example.com/quicklook.tif",
                    "type": "application/x-geotiff",
                    "title": "Quicklook COG",
                },
            },
        }
        asset = stac_widget._qlk_asset_from_item(item)
        assert asset is not None
        assert asset["href"] == "https://example.com/quicklook.tif"

    def test_returns_none_when_no_qlk_asset(self, stac_widget):
        """Should return None when item has no QLK asset."""
        item = {
            "id": "ICEYE_TEST_NO_QLK",
            "assets": {
                "slc-cog": {
                    "href": "https://example.com/item_SLC.tif",
                    "type": "application/x-geotiff",
                    "title": "SLC product",
                },
            },
        }
        asset = stac_widget._qlk_asset_from_item(item)
        assert asset is None

    def test_returns_none_when_assets_empty(self, stac_widget):
        """Should return None when assets dict is empty."""
        item = {"id": "ICEYE_EMPTY", "assets": {}}
        asset = stac_widget._qlk_asset_from_item(item)
        assert asset is None

    def test_returns_none_when_assets_missing(self, stac_widget):
        """Should return None when assets key is missing."""
        item = {"id": "ICEYE_NO_ASSETS"}
        asset = stac_widget._qlk_asset_from_item(item)
        assert asset is None


class TestOnItemsLoadedQlkBranching:
    """Tests for on_items_loaded branching: QLK vs footprint."""

    def test_launches_qlk_task_when_items_have_qlk(self, stac_widget):
        """Should launch StacItemsQlkTask when items have qlk-cog asset."""
        items = [
            {
                "id": "ICEYE_QLK_1",
                "assets": {
                    "qlk-cog": {
                        "href": "https://example.com/qlk1.tif",
                        "type": "application/x-geotiff",
                    },
                },
            },
        ]
        add_task_calls = []

        def capture_add_task(task):
            add_task_calls.append(task)

        with patch("iceye_toolbox.gui.stac_catalog_widget.QgsApplication") as mock_qgs:
            mock_qgs.taskManager.return_value.addTask.side_effect = capture_add_task
            stac_widget.on_items_loaded(True, items, None)

        assert len(add_task_calls) == 1
        assert isinstance(add_task_calls[0], StacItemsQlkTask)
        assert add_task_calls[0].items == items

    def test_adds_footprint_when_items_have_no_qlk(self, stac_widget):
        """Should add footprint via _add_item_layer when items lack QLK."""
        items = [
            {
                "id": "ICEYE_NO_QLK",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0.0, 0.0],
                            [1.0, 0.0],
                            [1.0, 1.0],
                            [0.0, 1.0],
                            [0.0, 0.0],
                        ]
                    ],
                },
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "assets": {"slc-cog": {"href": "https://example.com/slc.tif"}},
            },
        ]
        with patch.object(
            stac_widget, "_add_item_layer", return_value=True
        ) as mock_add:
            stac_widget.on_items_loaded(True, items, None)

        mock_add.assert_called_once_with(items[0])
        assert stac_widget.loadGeojsonButton.isEnabled()

    def test_mixed_items_splits_correctly(self, stac_widget):
        """Items with QLK go to task; items without go to footprint."""
        qlk_item = {
            "id": "ICEYE_WITH_QLK",
            "assets": {
                "qlk-cog": {
                    "href": "https://example.com/qlk.tif",
                    "type": "application/x-geotiff",
                },
            },
        }
        footprint_item = {
            "id": "ICEYE_NO_QLK",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
                ],
            },
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "assets": {},
        }
        items = [qlk_item, footprint_item]

        add_task_calls = []

        def capture_add_task(task):
            add_task_calls.append(task)

        with (
            patch("iceye_toolbox.gui.stac_catalog_widget.QgsApplication") as mock_qgs,
            patch.object(stac_widget, "_add_item_layer", return_value=True) as mock_add,
        ):
            mock_qgs.taskManager.return_value.addTask.side_effect = capture_add_task
            stac_widget.on_items_loaded(True, items, None)

        mock_add.assert_called_once_with(footprint_item)
        assert len(add_task_calls) == 1
        assert isinstance(add_task_calls[0], StacItemsQlkTask)
        assert add_task_calls[0].items == [qlk_item]


class TestQlkCacheDir:
    """Tests for _qlk_cache_dir helper."""

    def test_returns_path_under_iceye_toolbox(self, stac_widget):
        """Should return path ending with iceye_toolbox/qlk_cache."""
        path = stac_widget._qlk_cache_dir()
        assert "iceye_toolbox" in str(path)
        assert path.name == "qlk_cache"


# =============================================================================
# Parallel SLC Download Tests
# =============================================================================


class TestSlcFilenameFromUrl:
    """Tests for _slc_filename_from_url helper."""

    def test_extracts_filename_from_url_path(self):
        """Should extract filename from URL path."""
        url = "https://example.com/catalog/ICEYE_SLC_123.tif"
        assert _slc_filename_from_url(url, "ICEYE_123") == "ICEYE_SLC_123.tif"

    def test_uses_item_id_when_path_empty(self):
        """Should use item_id when path has no filename."""
        url = "https://example.com/"
        assert _slc_filename_from_url(url, "ICEYE_ITEM") == "ICEYE_ITEM.tif"

    def test_adds_tif_extension_when_missing(self):
        """Should add .tif when extension is missing."""
        url = "https://example.com/slc_file"
        assert _slc_filename_from_url(url, "item") == "slc_file.tif"

    def test_uses_slc_default_when_item_id_none(self):
        """Should use 'slc' as default when item_id is None."""
        url = "https://example.com/"
        assert _slc_filename_from_url(url, None) == "slc.tif"


class TestStacSlcSingleDownloadTask:
    """Tests for StacSlcSingleDownloadTask (one file per task)."""

    def test_skips_download_when_file_exists(self, tmp_path):
        """Should return success without downloading when file already exists."""
        existing_file = tmp_path / "ICEYE_SLC_123.tif"
        existing_file.write_bytes(b"existing content")
        parent_results = []

        task = StacSlcSingleDownloadTask(
            url="https://example.com/slc.tif",
            path=str(existing_file),
            item_id="ICEYE_123",
            parent_results=parent_results,
        )
        result = task.run()

        assert result is True
        assert task.result == {
            "id": "ICEYE_123",
            "path": str(existing_file),
            "error": None,
        }

    def test_finished_appends_result_to_parent(self, tmp_path):
        """Should append result to parent_results in finished()."""
        existing_file = tmp_path / "exists.tif"
        existing_file.write_bytes(b"x")
        parent_results = []

        task = StacSlcSingleDownloadTask(
            url="https://example.com/slc.tif",
            path=str(existing_file),
            item_id="ITEM_1",
            parent_results=parent_results,
        )
        task.run()
        task.finished(True)

        assert len(parent_results) == 1
        assert parent_results[0] == {
            "id": "ITEM_1",
            "path": str(existing_file),
            "error": None,
        }

    def test_download_success_with_mocked_urlopen(self, tmp_path):
        """Should download file when urlopen returns valid response."""
        dest = tmp_path / "downloaded.tif"
        parent_results = []

        mock_response = MagicMock()
        mock_response.read.side_effect = [b"chunk1", b"chunk2", b""]
        mock_response.getheader.return_value = "12"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        task = StacSlcSingleDownloadTask(
            url="https://example.com/slc.tif",
            path=str(dest),
            item_id="ICEYE_DL",
            parent_results=parent_results,
        )

        with patch(
            "iceye_toolbox.gui.stac_catalog_widget.urlopen",
            return_value=mock_response,
        ):
            result = task.run()

        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == b"chunk1chunk2"
        assert task.result["error"] is None


class TestStacSlcBatchTask:
    """Tests for StacSlcBatchTask (parent coordinator)."""

    def test_creates_one_subtask_per_download_item(self, stac_widget, tmp_path):
        """Should create one StacSlcSingleDownloadTask per download item."""
        file1 = tmp_path / "slc1.tif"
        file1.write_bytes(b"1")
        file2 = tmp_path / "slc2.tif"
        file2.write_bytes(b"2")

        download_items = [
            ("https://example.com/1.tif", str(file1), "ITEM_1"),
            ("https://example.com/2.tif", str(file2), "ITEM_2"),
        ]
        error_results = []

        task = StacSlcBatchTask(
            widget=stac_widget,
            download_items=download_items,
            error_results=error_results,
        )

        assert len(task.download_items) == 2

    def test_finished_aggregates_error_results_and_subtask_results(
        self, stac_widget, tmp_path
    ):
        """Should pass error_results + subtask results to on_slc_loaded."""
        file1 = tmp_path / "slc1.tif"
        file1.write_bytes(b"1")

        download_items = [("https://example.com/1.tif", str(file1), "ITEM_1")]
        error_results = [
            {"id": "ITEM_BAD", "path": None, "error": "No SLC asset found."},
        ]

        task = StacSlcBatchTask(
            widget=stac_widget,
            download_items=download_items,
            error_results=error_results,
        )

        with patch.object(stac_widget, "on_slc_loaded") as mock_on_slc:
            task.finished(True)

        mock_on_slc.assert_called_once()
        call_args = mock_on_slc.call_args[0]
        assert call_args[0] is True
        results = call_args[1]
        assert len(results) >= 1
        assert results[0] == error_results[0]


class TestLoadSelectedSlc:
    """Tests for load_selected_slc (parallel SLC download flow)."""

    def test_launches_batch_task_with_multiple_items(self, stac_widget, tmp_path):
        """Should launch StacSlcBatchTask with multiple download items."""
        stac_item_1 = {
            "id": "ICEYE_SLC_1",
            "assets": {
                "slc-cog": {
                    "href": "https://example.com/slc1.tif",
                    "type": "application/x-geotiff",
                },
            },
        }
        stac_item_2 = {
            "id": "ICEYE_SLC_2",
            "assets": {
                "slc-cog": {
                    "href": "https://example.com/slc2.tif",
                    "type": "application/x-geotiff",
                },
            },
        }

        parent = QTreeWidgetItem(["Collection"])
        child1 = QTreeWidgetItem(["ICEYE_SLC_1"])
        child1.setData(0, Qt.UserRole, "https://example.com/item1.json")
        child2 = QTreeWidgetItem(["ICEYE_SLC_2"])
        child2.setData(0, Qt.UserRole, "https://example.com/item2.json")
        parent.addChild(child1)
        parent.addChild(child2)
        stac_widget.catalogTree.addTopLevelItem(parent)
        child1.setSelected(True)
        child2.setSelected(True)

        add_task_calls = []

        def capture_add_task(task):
            add_task_calls.append(task)

        with (
            patch("iceye_toolbox.gui.stac_catalog_widget.QgsApplication") as mock_qgs,
            patch(
                "iceye_toolbox.gui.stac_catalog_widget.stac_client.fetch_item",
                side_effect=[stac_item_1, stac_item_2],
            ),
            patch.object(stac_widget, "_slc_cache_dir", return_value=tmp_path),
        ):
            mock_qgs.taskManager.return_value.addTask.side_effect = capture_add_task
            stac_widget.load_selected_slc()

        assert len(add_task_calls) == 1
        assert isinstance(add_task_calls[0], StacSlcBatchTask)
        batch = add_task_calls[0]
        assert len(batch.download_items) == 2
        assert batch.download_items[0][2] == "ICEYE_SLC_1"
        assert batch.download_items[1][2] == "ICEYE_SLC_2"

    def test_adds_error_result_when_no_slc_asset(self, stac_widget, tmp_path):
        """Should add error result when item has no SLC asset."""
        stac_item_no_slc = {
            "id": "ICEYE_NO_SLC",
            "assets": {"qlk-cog": {"href": "https://example.com/qlk.tif"}},
        }

        parent = QTreeWidgetItem(["Collection"])
        child = QTreeWidgetItem(["ICEYE_NO_SLC"])
        child.setData(0, Qt.UserRole, "https://example.com/item.json")
        parent.addChild(child)
        stac_widget.catalogTree.addTopLevelItem(parent)
        child.setSelected(True)

        add_task_calls = []

        def capture_add_task(task):
            add_task_calls.append(task)

        with (
            patch("iceye_toolbox.gui.stac_catalog_widget.QgsApplication") as mock_qgs,
            patch(
                "iceye_toolbox.gui.stac_catalog_widget.stac_client.fetch_item",
                return_value=stac_item_no_slc,
            ),
            patch.object(stac_widget, "_slc_cache_dir", return_value=tmp_path),
        ):
            mock_qgs.taskManager.return_value.addTask.side_effect = capture_add_task
            stac_widget.load_selected_slc()

        assert len(add_task_calls) == 1
        batch = add_task_calls[0]
        assert len(batch.download_items) == 0
        assert len(batch.error_results) == 1
        assert batch.error_results[0]["error"] == "No SLC asset found."

    def test_does_not_launch_when_no_selection(self, stac_widget):
        """Should not launch task when no items selected."""
        add_task_calls = []

        def capture_add_task(task):
            add_task_calls.append(task)

        with patch("iceye_toolbox.gui.stac_catalog_widget.QgsApplication") as mock_qgs:
            mock_qgs.taskManager.return_value.addTask.side_effect = capture_add_task
            stac_widget.load_selected_slc()

        assert len(add_task_calls) == 0

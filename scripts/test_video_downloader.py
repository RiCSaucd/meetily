#!/usr/bin/env python3
"""
Unit tests for video_downloader.py.

Runs without network access or yt-dlp installed: a fake yt_dlp module is
injected so the tests exercise parameter validation, mode-specific option
building, and playlist gating.

Usage:
    python -m unittest test_video_downloader -v
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import video_downloader


class FakeDownloadError(Exception):
    pass


class FakeYoutubeDL:
    """Records the opts it was built with and replays canned info dicts."""

    # Class-level queues configured per test.
    probe_info = None
    download_info = None
    instances = []

    def __init__(self, opts):
        self.opts = opts
        FakeYoutubeDL.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        return (
            FakeYoutubeDL.download_info
            if download
            else FakeYoutubeDL.probe_info
        )


class FakeYtDlp:
    YoutubeDL = FakeYoutubeDL

    class utils:
        DownloadError = FakeDownloadError


def single_video_info(**overrides):
    info = {
        "id": "abc123",
        "title": "Some Video",
        "duration": 61,
        "webpage_url": "https://youtube.com/watch?v=abc123",
        "requested_downloads": [{"filepath": "/out/Some Video [abc123].mp4"}],
    }
    info.update(overrides)
    return info


def playlist_info(entries):
    return {
        "_type": "playlist",
        "id": "PL123",
        "title": "Some Playlist",
        "entries": entries,
    }


class VideoDownloaderTestCase(unittest.TestCase):
    def setUp(self):
        FakeYoutubeDL.instances = []
        FakeYoutubeDL.probe_info = single_video_info()
        FakeYoutubeDL.download_info = single_video_info()

        patcher = mock.patch.object(
            video_downloader, "_load_yt_dlp", return_value=FakeYtDlp
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_dir = Path(self._tmp.name) / "assets" / "reference"

    def base_params(self, **overrides):
        params = {
            "url": "https://youtube.com/watch?v=abc123",
            "output_dir": str(self.output_dir),
        }
        params.update(overrides)
        return params

    def download_opts(self):
        """Opts passed to the download-phase YoutubeDL (second instance)."""
        self.assertEqual(len(FakeYoutubeDL.instances), 2)
        return FakeYoutubeDL.instances[1].opts


class ValidationTests(VideoDownloaderTestCase):
    def test_missing_url_rejected(self):
        with self.assertRaisesRegex(ValueError, "'url' is required"):
            video_downloader.execute({"output_dir": str(self.output_dir)})

    def test_non_http_url_rejected(self):
        with self.assertRaisesRegex(ValueError, "http\\(s\\) URL"):
            video_downloader.execute(
                self.base_params(url="file:///etc/passwd")
            )

    def test_missing_output_dir_rejected(self):
        with self.assertRaisesRegex(ValueError, "'output_dir' is required"):
            video_downloader.execute({"url": "https://example.com/v"})

    def test_unknown_parameter_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown parameter.*playlist"):
            video_downloader.execute(self.base_params(playlist=True))

    def test_invalid_ingest_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "'ingest_mode'"):
            video_downloader.execute(self.base_params(ingest_mode="archive"))

    def test_non_boolean_allow_playlist_rejected(self):
        with self.assertRaisesRegex(ValueError, "'allow_playlist'"):
            video_downloader.execute(self.base_params(allow_playlist="yes"))

    def test_non_positive_max_playlist_items_rejected(self):
        with self.assertRaisesRegex(ValueError, "'max_playlist_items'"):
            video_downloader.execute(
                self.base_params(
                    allow_playlist=True, max_playlist_items=0
                )
            )

    def test_boolean_max_playlist_items_rejected(self):
        with self.assertRaisesRegex(ValueError, "'max_playlist_items'"):
            video_downloader.execute(
                self.base_params(
                    allow_playlist=True, max_playlist_items=True
                )
            )


class ReferenceModeTests(VideoDownloaderTestCase):
    def test_defaults_to_reference_mode(self):
        result = video_downloader.execute(self.base_params())
        self.assertEqual(result["ingest_mode"], "reference")

    def test_reference_mode_caps_resolution_and_blocks_playlists(self):
        video_downloader.execute(self.base_params())
        opts = self.download_opts()
        self.assertEqual(opts["format"], video_downloader.REFERENCE_FORMAT)
        self.assertTrue(opts["noplaylist"])
        self.assertNotIn("playlist_items", opts)

    def test_reference_mode_writes_metadata_sidecar(self):
        video_downloader.execute(self.base_params())
        self.assertTrue(self.download_opts()["writeinfojson"])

    def test_creates_output_dir_and_reports_download(self):
        result = video_downloader.execute(self.base_params())
        self.assertTrue(self.output_dir.is_dir())
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["is_playlist"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(
            result["downloads"],
            [
                {
                    "id": "abc123",
                    "title": "Some Video",
                    "duration": 61,
                    "webpage_url": "https://youtube.com/watch?v=abc123",
                    "filepath": "/out/Some Video [abc123].mp4",
                }
            ],
        )

    def test_playlist_url_rejected_without_allow_playlist(self):
        FakeYoutubeDL.probe_info = playlist_info([])
        with self.assertRaisesRegex(ValueError, "allow_playlist"):
            video_downloader.execute(
                self.base_params(url="https://youtube.com/playlist?list=PL123")
            )
        # Nothing downloaded and no directory created for a rejected request.
        self.assertFalse(self.output_dir.exists())
        self.assertEqual(len(FakeYoutubeDL.instances), 1)


class ProductionModeTests(VideoDownloaderTestCase):
    def production_params(self, **overrides):
        return self.base_params(
            url="https://youtube.com/playlist?list=PL123",
            ingest_mode="production",
            allow_playlist=True,
            max_playlist_items=10,
            **overrides,
        )

    def setUp(self):
        super().setUp()
        FakeYoutubeDL.probe_info = playlist_info([{"id": "e1"}, {"id": "e2"}])
        FakeYoutubeDL.download_info = playlist_info(
            [
                single_video_info(id="e1", title="Clip 1"),
                single_video_info(id="e2", title="Clip 2"),
            ]
        )

    def test_production_playlist_options(self):
        video_downloader.execute(self.production_params())
        opts = self.download_opts()
        self.assertEqual(opts["format"], video_downloader.PRODUCTION_FORMAT)
        self.assertFalse(opts["noplaylist"])
        self.assertEqual(opts["playlist_items"], "1:10")
        self.assertEqual(opts["ignoreerrors"], "only_download")
        self.assertNotIn("writeinfojson", opts)

    def test_playlist_result_lists_all_entries(self):
        result = video_downloader.execute(self.production_params())
        self.assertTrue(result["is_playlist"])
        self.assertEqual(result["playlist_title"], "Some Playlist")
        self.assertEqual(
            [d["title"] for d in result["downloads"]], ["Clip 1", "Clip 2"]
        )
        self.assertEqual(result["errors"], [])

    def test_failed_playlist_entries_reported_as_errors(self):
        FakeYoutubeDL.download_info = playlist_info(
            [single_video_info(id="e1", title="Clip 1"), None]
        )
        result = video_downloader.execute(self.production_params())
        self.assertEqual(len(result["downloads"]), 1)
        self.assertEqual(
            result["errors"], ["Playlist entry 2 failed to download"]
        )

    def test_single_video_in_production_mode(self):
        FakeYoutubeDL.probe_info = single_video_info()
        FakeYoutubeDL.download_info = single_video_info()
        result = video_downloader.execute(
            self.base_params(ingest_mode="production")
        )
        self.assertFalse(result["is_playlist"])
        opts = self.download_opts()
        self.assertEqual(opts["format"], video_downloader.PRODUCTION_FORMAT)
        self.assertTrue(opts["noplaylist"])

    def test_probe_error_wrapped_as_runtime_error(self):
        with mock.patch.object(
            FakeYoutubeDL,
            "extract_info",
            side_effect=FakeDownloadError("403 Forbidden"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Could not resolve"):
                video_downloader.execute(self.production_params())

    def test_download_error_wrapped_as_runtime_error(self):
        def boom(url, download=False):
            if download:
                raise FakeDownloadError("HTTP Error 403")
            return FakeYoutubeDL.probe_info

        with mock.patch.object(FakeYoutubeDL, "extract_info", autospec=True) as m:
            m.side_effect = lambda self, url, download=False: boom(
                url, download
            )
            with self.assertRaisesRegex(RuntimeError, "Download failed"):
                video_downloader.execute(self.production_params())


if __name__ == "__main__":
    unittest.main()

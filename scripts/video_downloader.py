#!/usr/bin/env python3
"""
Video Downloader Tool

Downloads videos (YouTube and other yt-dlp supported sites) into a project
directory with two ingest modes:

- "reference" (default): quick single-video grabs for reference analysis.
  Resolution is capped at 720p to keep downloads fast, a metadata sidecar
  (.info.json) is written next to the video, and playlist expansion is
  disabled unless explicitly allowed.

- "production": full-quality ingest for downstream pipelines (e.g. the
  clip-factory workflow). Downloads the best available video+audio and can
  ingest playlists when `allow_playlist` is set, bounded by
  `max_playlist_items`.

Programmatic usage:

    import video_downloader

    # Reference analysis (default)
    video_downloader.execute({
        "url": "https://youtube.com/watch?v=...",
        "output_dir": "projects/my-project/assets/reference",
    })

    # Production playlist ingest for clip-factory
    video_downloader.execute({
        "url": "https://youtube.com/playlist?list=...",
        "output_dir": "projects/my-project/assets/source",
        "ingest_mode": "production",
        "allow_playlist": True,
        "max_playlist_items": 10,
    })

CLI usage:

    python video_downloader.py URL --output-dir DIR
    python video_downloader.py URL --output-dir DIR \
        --ingest-mode production --allow-playlist --max-playlist-items 10

Requires yt-dlp: pip install yt-dlp
"""

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

INGEST_MODE_REFERENCE = "reference"
INGEST_MODE_PRODUCTION = "production"
INGEST_MODES = (INGEST_MODE_REFERENCE, INGEST_MODE_PRODUCTION)

DEFAULT_MAX_PLAYLIST_ITEMS = 25

# Reference mode caps resolution for fast analysis downloads.
REFERENCE_FORMAT = (
    "bestvideo[height<=720]+bestaudio/best[height<=720]/best"
)
# Production mode takes the best available quality for ingest pipelines.
PRODUCTION_FORMAT = "bestvideo+bestaudio/best"

SINGLE_OUTTMPL = "%(title)s [%(id)s].%(ext)s"
PLAYLIST_OUTTMPL = "%(playlist_index)03d - %(title)s [%(id)s].%(ext)s"

KNOWN_PARAMS = {
    "url",
    "output_dir",
    "ingest_mode",
    "allow_playlist",
    "max_playlist_items",
}


def _load_yt_dlp():
    """Import yt-dlp lazily so validation and tests work without it."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is required for video downloads. "
            "Install it with: pip install yt-dlp"
        ) from exc
    return yt_dlp


def _validate_params(params: dict) -> dict:
    """Validate and normalize the request params for execute()."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    unknown = set(params) - KNOWN_PARAMS
    if unknown:
        raise ValueError(
            f"Unknown parameter(s): {', '.join(sorted(unknown))}. "
            f"Supported parameters: {', '.join(sorted(KNOWN_PARAMS))}"
        )

    url = params.get("url")
    if not url or not isinstance(url, str):
        raise ValueError("'url' is required and must be a string")
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"'url' must be an http(s) URL, got: {url}")

    output_dir = params.get("output_dir")
    if not output_dir or not isinstance(output_dir, (str, Path)):
        raise ValueError("'output_dir' is required and must be a path string")

    ingest_mode = params.get("ingest_mode", INGEST_MODE_REFERENCE)
    if ingest_mode not in INGEST_MODES:
        raise ValueError(
            f"'ingest_mode' must be one of {INGEST_MODES}, got: {ingest_mode!r}"
        )

    allow_playlist = params.get("allow_playlist", False)
    if not isinstance(allow_playlist, bool):
        raise ValueError("'allow_playlist' must be a boolean")

    max_playlist_items = params.get(
        "max_playlist_items", DEFAULT_MAX_PLAYLIST_ITEMS
    )
    if (
        isinstance(max_playlist_items, bool)
        or not isinstance(max_playlist_items, int)
        or max_playlist_items < 1
    ):
        raise ValueError("'max_playlist_items' must be a positive integer")

    return {
        "url": url,
        "output_dir": Path(output_dir),
        "ingest_mode": ingest_mode,
        "allow_playlist": allow_playlist,
        "max_playlist_items": max_playlist_items,
    }


def _probe_url(yt_dlp, url: str, allow_playlist: bool, max_items: int) -> dict:
    """Classify the URL (single video vs playlist) without downloading."""
    probe_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        # Flat extraction avoids resolving every playlist entry up front.
        "extract_flat": "in_playlist",
        # For watch?v=...&list=... URLs, ignore the playlist part unless
        # playlists were explicitly allowed.
        "noplaylist": not allow_playlist,
    }
    if allow_playlist:
        probe_opts["playlist_items"] = f"1:{max_items}"

    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(f"Could not resolve {url}: {exc}") from exc
    if info is None:
        raise RuntimeError(f"Could not extract any information from: {url}")
    return info


def _build_download_opts(opts: dict, is_playlist: bool, quiet: bool) -> dict:
    """Build yt-dlp options for the actual download."""
    outtmpl = PLAYLIST_OUTTMPL if is_playlist else SINGLE_OUTTMPL
    ydl_opts = {
        "format": (
            PRODUCTION_FORMAT
            if opts["ingest_mode"] == INGEST_MODE_PRODUCTION
            else REFERENCE_FORMAT
        ),
        "outtmpl": str(opts["output_dir"] / outtmpl),
        "merge_output_format": "mp4",
        "noplaylist": not opts["allow_playlist"],
        "quiet": quiet,
        "no_warnings": quiet,
        "noprogress": quiet,
    }
    if opts["ingest_mode"] == INGEST_MODE_REFERENCE:
        # Reference downloads are for analysis: keep the metadata sidecar.
        ydl_opts["writeinfojson"] = True
    if is_playlist:
        ydl_opts["playlist_items"] = f"1:{opts['max_playlist_items']}"
        # One broken playlist entry should not abort the whole ingest.
        ydl_opts["ignoreerrors"] = "only_download"
    return ydl_opts


def _summarize_entry(entry: dict) -> dict:
    """Extract the fields callers care about from a yt-dlp info dict."""
    filepath = None
    for download in entry.get("requested_downloads") or []:
        if download.get("filepath"):
            filepath = download["filepath"]
            break
    return {
        "id": entry.get("id"),
        "title": entry.get("title"),
        "duration": entry.get("duration"),
        "webpage_url": entry.get("webpage_url"),
        "filepath": filepath,
    }


def execute(params: dict, quiet: bool = True) -> dict:
    """
    Download a video or playlist according to the requested ingest mode.

    Returns a result dict:
        {
            "status": "ok",
            "ingest_mode": "reference" | "production",
            "url": str,
            "output_dir": str (absolute),
            "is_playlist": bool,
            "playlist_title": str | None,
            "downloads": [{"id", "title", "duration", "webpage_url",
                           "filepath"}],
            "errors": [str],
        }

    Raises ValueError for invalid parameters (including playlist URLs when
    `allow_playlist` is not set) and RuntimeError when the download fails
    entirely.
    """
    opts = _validate_params(params)
    yt_dlp = _load_yt_dlp()

    probe = _probe_url(
        yt_dlp, opts["url"], opts["allow_playlist"], opts["max_playlist_items"]
    )
    is_playlist = probe.get("_type") == "playlist"
    if is_playlist and not opts["allow_playlist"]:
        raise ValueError(
            f"URL resolves to a playlist ({probe.get('title') or opts['url']}) "
            "but 'allow_playlist' is not set. Pass 'allow_playlist': True "
            "(and optionally 'max_playlist_items') to ingest playlists."
        )

    opts["output_dir"].mkdir(parents=True, exist_ok=True)

    ydl_opts = _build_download_opts(opts, is_playlist, quiet)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(opts["url"], download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(f"Download failed for {opts['url']}: {exc}") from exc
    if info is None:
        raise RuntimeError(f"Download produced no result for: {opts['url']}")

    downloads = []
    errors = []
    if info.get("_type") == "playlist":
        entries = list(info.get("entries") or [])
        for index, entry in enumerate(entries, start=1):
            if entry is None:
                errors.append(f"Playlist entry {index} failed to download")
            else:
                downloads.append(_summarize_entry(entry))
    else:
        downloads.append(_summarize_entry(info))

    return {
        "status": "ok",
        "ingest_mode": opts["ingest_mode"],
        "url": opts["url"],
        "output_dir": str(opts["output_dir"].resolve()),
        "is_playlist": is_playlist,
        "playlist_title": info.get("title") if is_playlist else None,
        "downloads": downloads,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download videos for reference analysis or production ingest"
    )
    parser.add_argument("url", help="Video or playlist URL")
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory to download into (created if missing)",
    )
    parser.add_argument(
        "--ingest-mode",
        choices=INGEST_MODES,
        default=INGEST_MODE_REFERENCE,
        help="reference: fast 720p-capped analysis download (default); "
        "production: best-quality ingest",
    )
    parser.add_argument(
        "--allow-playlist",
        action="store_true",
        help="Allow playlist URLs to expand and download multiple items",
    )
    parser.add_argument(
        "--max-playlist-items",
        type=int,
        default=DEFAULT_MAX_PLAYLIST_ITEMS,
        help="Maximum playlist entries to download "
        f"(default: {DEFAULT_MAX_PLAYLIST_ITEMS})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON instead of a summary",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show yt-dlp download progress and warnings",
    )
    args = parser.parse_args(argv)

    try:
        result = execute(
            {
                "url": args.url,
                "output_dir": args.output_dir,
                "ingest_mode": args.ingest_mode,
                "allow_playlist": args.allow_playlist,
                "max_playlist_items": args.max_playlist_items,
            },
            quiet=not args.verbose,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"Downloaded {len(result['downloads'])} item(s) "
            f"[{result['ingest_mode']} mode] to {result['output_dir']}"
        )
        for item in result["downloads"]:
            print(f"  - {item['title']}: {item['filepath']}")
        for error in result["errors"]:
            print(f"  ! {error}", file=sys.stderr)

    return 1 if result["errors"] and not result["downloads"] else 0


if __name__ == "__main__":
    sys.exit(main())

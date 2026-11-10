"""Temporary storage for files someone is putting together. Each upload gets its own folder on disk.

A folder is deleted when it expires or when newer uploads push it out, so nothing is kept for long.
"""
from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path, PurePosixPath

from .recipe_ops import RecipeError

MAX_FILES = 20
MAX_SOURCE_BYTES = 100 * 1024 * 1024
MAX_SOURCES = 10
MAX_AGE_SECONDS = 6 * 60 * 60
SUPPORTED_SUFFIXES = {".csv", ".xlsx"}
SNAPSHOT_FOLDER = "snapshots"


def clean_file_name(name: str) -> str:
    """Just the file's own name, with any folders removed, and only for file types the app can read."""
    base = PurePosixPath((name or "").replace("\\", "/")).name
    if PurePosixPath(base).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise RecipeError(f"{base or 'That file'} isn't a type this app can read. Save it as .csv or .xlsx and try again.")
    return base


class SourceStore:
    def __init__(self) -> None:
        self._items: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()
        self._root = Path(tempfile.mkdtemp(prefix="databoarder_sources_"))

    def _remove(self, source_id: str) -> None:
        item = self._items.pop(source_id, None)
        if item:
            shutil.rmtree(item["folder"], ignore_errors=True)

    def _remove_expired(self) -> None:
        cutoff = time.time() - MAX_AGE_SECONDS
        for source_id in [key for key, item in self._items.items() if item["created"] < cutoff]:
            self._remove(source_id)

    def add(self, files: list[tuple[str, bytes]]) -> str:
        """Saves one batch of files. Returns the id used to refer to them."""
        if not files:
            raise RecipeError("Please choose at least one file.")
        if len(files) > MAX_FILES:
            raise RecipeError(f"That's {len(files)} files, and the limit is {MAX_FILES}. Combine some of them and try again.")
        if sum(len(raw) for _, raw in files) > MAX_SOURCE_BYTES:
            raise RecipeError(f"Together these files are larger than {MAX_SOURCE_BYTES // (1024 * 1024)} MB. "
                              "Ask your analyst for a smaller pull, such as only the campaign's dates.")
        names = [clean_file_name(name) for name, _ in files]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise RecipeError(f"More than one file is called {', '.join(repeated)}. Rename one and try again.")
        source_id = uuid.uuid4().hex[:12]
        folder = self._root / source_id
        (folder / SNAPSHOT_FOLDER).mkdir(parents=True)
        for name, (_, raw) in zip(names, files):
            (folder / name).write_bytes(raw)
        with self._lock:
            self._remove_expired()
            while len(self._items) >= MAX_SOURCES:
                self._remove(next(iter(self._items)))
            self._items[source_id] = {"folder": folder, "created": time.time(), "names": names}
        return source_id

    def folder(self, source_id: str) -> Path:
        with self._lock:
            self._remove_expired()
            item = self._items.get(source_id)
        if not item:
            raise RecipeError("Those files are no longer loaded. Please upload them again.")
        return item["folder"]

    def paths(self, source_id: str) -> list[Path]:
        folder = self.folder(source_id)
        return [folder / name for name in self._items[source_id]["names"]]

    def snapshot_path(self, source_id: str, dataset_id: str) -> Path:
        return self.folder(source_id) / SNAPSHOT_FOLDER / f"{dataset_id}.csv"

    def discard(self, source_id: str) -> None:
        with self._lock:
            self._remove(source_id)

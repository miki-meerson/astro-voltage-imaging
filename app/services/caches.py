"""All in-process caches of the web app, in one place.

Every cache is a small thread-safe LRU (``LRUCache``): bounded by the number of entries, one lock per cache.
Before the refactor several of these were capacity-1 dicts (``cache.clear(); cache[key] = value``), so two open
experiments or browser tabs evicted each other on every request; the bounds below keep the memory use small
but let a few keys live side by side.
"""
from __future__ import annotations

from collections import OrderedDict
import threading


class LRUCache:
    """Least-recently-used cache with a fixed number of entries; all operations hold an internal lock."""

    def __init__(self, max_entries: int, name: str = "") -> None:
        self.max_entries = int(max_entries)
        self.name = name
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            if key not in self._data:
                return default
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key, value) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


STA_SIGNIFICANT = LRUCache(8, "sta_significant")        # key: (run_id, csv mtime_ns, csv size)
BEHAVIOR_PAYLOAD = LRUCache(8, "behavior_payload")      # key: (run_id, npz mtime_ns)
MOVIE_META = LRUCache(16, "movie_meta")                 # key: (path, mtime_ns)
MOVIE_ARRAY = LRUCache(3, "movie_array")                # holds np.memmap objects (open file handles): keep it tiny
MOVIE_JPEG = LRUCache(160, "movie_jpeg")                # key: (path, mtime_ns, frame, max_size, contrast, ...) -> bytes
MEAN_INTENSITY = LRUCache(8, "mean_intensity")
ANATOMY_CROP_INFO = LRUCache(16, "anatomy_crop_info")
STA_STACK = LRUCache(6, "sta_stack")                    # key -> binary payload of the cropped STA movie
CELL_REVIEW = LRUCache(16, "cell_review")               # key: (experiment, spike run, max mtime of label files and iscell.npy)
STA_SUMMARY_PNG = LRUCache(80, "sta_summary_png")       # rendered cell-page STA summary images, bytes
CLUSTER_RESULT = LRUCache(240, "cluster_result")        # key: (npz path, mtime_ns) -> parsed cluster result


def clear_movie_caches() -> None:
    """Drop everything that may hold a handle on, or a rendering of, a movie file (before the movie is rewritten)."""
    MOVIE_ARRAY.clear()
    MOVIE_META.clear()
    MOVIE_JPEG.clear()
    MEAN_INTENSITY.clear()


def clear_experiment_result_caches() -> None:
    """Drop the summaries derived from the STA / spike / behavior results of an experiment."""
    STA_SIGNIFICANT.clear()
    BEHAVIOR_PAYLOAD.clear()
    CELL_REVIEW.clear()

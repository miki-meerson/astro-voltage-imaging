"""The hypothesis catalogue: which STA analyses are run on an experiment, each with its own analysis window.

A hypothesis is a named set of settings (how long before and after the spike, where the baseline is, which part is tested). The catalogue is a
yml file in the project folder (hypotheses.yml; hypotheses.example.yml when you have no own copy). It never decides which result types run:
that is chosen by the user for each run. Everything is checked when the file is read, and every setting must be given explicitly.

Settings are seconds relative to the spike peak (negative = before). The settings of a hypothesis are copied into each run when the run is
created, so editing the catalogue later does not change finished results.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from utils import globals_utils as g
from utils.paths_utils import PathKind, path_for

STATISTICS = ("baseline_subtracted",)  # the statistics the STA job can compute (more are added with the hypotheses that need them)
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")  # safe in file names, step names and urls
WINDOW_SETTINGS = ("baseline_window_seconds", "tested_window_seconds")
NUMBER_SETTINGS = ("pre_seconds", "post_seconds", "silence_before_seconds", "silence_after_seconds", "time_bin_seconds")
SETTINGS = ("name", "description", *NUMBER_SETTINGS, *WINDOW_SETTINGS, "statistic")


@dataclass(frozen=True)
class Window:
    """A stretch of time [start, end] in seconds relative to the spike (negative = before)."""
    start: float
    end: float

    def inside(self, other: Window) -> bool:
        return other.start <= self.start and self.end <= other.end

    def ends_before(self, other: Window) -> bool:
        return self.end < other.start

    def __str__(self) -> str:
        return f"[{self.start:g}, {self.end:g}]"


@dataclass(frozen=True)
class Hypothesis:
    name: str
    description: str
    pre_seconds: float  # saved window before the spike
    post_seconds: float  # saved window after the spike
    baseline_window_seconds: tuple[float, float]  # the baseline: its median normalizes each spike (F0) and its average is subtracted from the average (baseline_subtracted)
    tested_window_seconds: tuple[float, float]  # the part that is tested against the chance copies
    silence_before_seconds: float  # a spike is used only if no other spike of the cell is this close before it (0 = no condition)
    silence_after_seconds: float  # ... and none this close after it (0 = no condition); the random frames follow the same rule
    time_bin_seconds: float  # the tested window is averaged in bins of this length (0 = every frame is tested)
    statistic: str

    def settings(self) -> dict:
        """The settings as stored with a run (plain numbers and lists)."""
        return {
            "hypothesis": self.name,
            "pre_seconds": self.pre_seconds,
            "post_seconds": self.post_seconds,
            "baseline_window_seconds": list(self.baseline_window_seconds),
            "tested_window_seconds": list(self.tested_window_seconds),
            "silence_before_seconds": self.silence_before_seconds,
            "silence_after_seconds": self.silence_after_seconds,
            "time_bin_seconds": self.time_bin_seconds,
            "statistic": self.statistic,
        }


RUN_SETTINGS = ("hypothesis", *NUMBER_SETTINGS, *WINDOW_SETTINGS, "statistic")  # the keys of Hypothesis.settings(), as stored in a run's parameters


def run_settings(params: dict) -> dict:
    """The hypothesis settings of a run's parameters (the keys of Hypothesis.settings()). Raises a ValueError naming the missing ones: there are no defaults."""
    missing = [key for key in RUN_SETTINGS if key not in params]
    if missing:
        raise ValueError(
            f"The analysis has no hypothesis setting(s) {', '.join(missing)}. Start the analysis from the experiment page (Run analysis): "
            "it copies the settings of a hypothesis into the run."
        )
    return {key: params[key] for key in RUN_SETTINGS}


def seconds_to_frames(seconds: float, fps: float) -> int:
    """Seconds as a number of frames (rounded to the nearest frame)."""
    return int(round(float(seconds) * float(fps)))


def time_bin_frames(settings: dict, fps: float) -> int:
    """Frames per time bin of the tested window (1 = no binning: every frame is tested)."""
    return max(1, seconds_to_frames(run_settings(settings)["time_bin_seconds"], fps))


def frame_windows(settings: dict, fps: float) -> tuple:
    """The hypothesis settings (seconds) as frames: (pre, post, pre_window, post_window, analysis_window, baseline_window).

    pre / post: saved frames before / after the spike. analysis_window: the tested window; baseline_window: the baseline (F0 and the subtracted average); pre_window = (tested start, -1) and post_window = (0, tested end) are
    the two halves averaged for the post-minus-pre map. All windows are frames relative to the spike."""
    settings = run_settings(settings)
    pre, post = seconds_to_frames(settings["pre_seconds"], fps), seconds_to_frames(settings["post_seconds"], fps)
    tested = tuple(seconds_to_frames(s, fps) for s in settings["tested_window_seconds"])
    baseline = tuple(seconds_to_frames(s, fps) for s in settings["baseline_window_seconds"])
    return pre, post, (tested[0], -1), (0, tested[1]), tested, baseline


def sta_step(name: str) -> str:
    """The pipeline step of the hypothesis `name` (for example 'sta:fast')."""
    return g.STA_STEP_PREFIX + name


def is_sta_step(step: str) -> bool:
    return str(step).startswith(g.STA_STEP_PREFIX)


def hypothesis_name(step: str) -> str:
    """The hypothesis name inside an STA step ('sta:fast' -> 'fast')."""
    if not is_sta_step(step):
        raise ValueError(f"{step!r} is not an STA step (STA steps start with {g.STA_STEP_PREFIX!r}).")
    return str(step)[len(g.STA_STEP_PREFIX):]


def catalogue_file(project_root: str | Path) -> Path:
    """The user's hypotheses.yml when it exists, else the tracked hypotheses.example.yml."""
    own = path_for(PathKind.HYPOTHESES_FILE, project_root)
    return own if own.exists() else path_for(PathKind.HYPOTHESES_EXAMPLE_FILE, project_root)


def _number(entry: dict, key: str, where: str) -> float:
    value = entry[key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"{where}: {key} must be a number, got {value!r}.")
    return float(value)


def _window(entry: dict, key: str, where: str) -> tuple[float, float]:
    value = entry[key]
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(isinstance(v, (int, float)) for v in value):
        raise ValueError(f"{where}: {key} must be two numbers [start, end], got {value!r}.")
    return float(value[0]), float(value[1])


def _require(where: str, rules: list[tuple[bool, str]]) -> None:
    """Every rule is (holds, what is wrong when it does not). All broken rules are reported together."""
    problems = [message for holds, message in rules if not holds]
    if problems:
        raise ValueError(f"{where}: " + "; ".join(problems) + ".")


def _hypothesis(entry: object, index: int, source: Path) -> Hypothesis:
    where = f"{source.name}, entry {index + 1}"
    missing = [key for key in SETTINGS if key not in entry]
    _require(where, [(not missing, f"missing setting(s) {', '.join(missing)}")])
    name = entry["name"]
    _require(where, [
        (isinstance(name, str) and bool(NAME_PATTERN.match(name)),
         f"the name {name!r} must be lower case letters, digits and underscores, starting with a letter (at most 32 characters)"),
        (entry["statistic"] in STATISTICS, f"unknown statistic {entry['statistic']!r}. Available: {', '.join(STATISTICS)}"),
    ])
    where = f"{source.name}, hypothesis {entry['name']}"
    hypothesis = Hypothesis(
        name=entry["name"], description=str(entry["description"]).strip(), statistic=entry["statistic"],
        **{key: _number(entry, key, where) for key in NUMBER_SETTINGS}, **{key: _window(entry, key, where) for key in WINDOW_SETTINGS},
    )
    saved = Window(-hypothesis.pre_seconds, hypothesis.post_seconds)
    base, tested = Window(*hypothesis.baseline_window_seconds), Window(*hypothesis.tested_window_seconds)
    _require(where, [
        (base.start < base.end, f"baseline_window_seconds {base} must start before it ends"),
        (tested.start < tested.end, f"tested_window_seconds {tested} must start before it ends"),
        (base.inside(saved), f"baseline_window_seconds {base} must lie inside the saved window {saved}"),
        (tested.inside(saved), f"tested_window_seconds {tested} must lie inside the saved window {saved}"),
        (base.ends_before(tested), f"the baseline window must end before the tested window starts (baseline {base}, tested {tested})"),
        (hypothesis.time_bin_seconds >= 0, "time_bin_seconds must be 0 (no binning) or more"),
        (hypothesis.time_bin_seconds == 0 or hypothesis.time_bin_seconds <= tested.end - tested.start, f"time_bin_seconds {hypothesis.time_bin_seconds:g} is longer than the tested window {tested}"),
    ])
    return hypothesis


def load_hypotheses(source: str | Path) -> tuple[Hypothesis, ...]:
    """The hypotheses of a catalogue file, in file order. Raises ValueError (naming the file and the entry) when anything is wrong."""
    source = Path(source)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{source.name} is not valid yml: {exc}") from exc
    entries = data.get("hypotheses") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError(f"{source.name} must have a non-empty list called 'hypotheses', each entry a mapping of settings.")
    hypotheses = tuple(_hypothesis(entry, i, source) for i, entry in enumerate(entries))
    names = [h.name for h in hypotheses]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"{source.name}: hypothesis name(s) used twice: {', '.join(repeated)}.")
    return hypotheses

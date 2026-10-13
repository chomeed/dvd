"""Critical-state annotations for MimicGen Coffee, derived from the simulator's own signals."""

from __future__ import annotations

from dataclasses import dataclass, field


STAGES: tuple[str, ...] = ("grasp", "rim", "insertion", "task")


LADDERS = {
    "gated": (
        ("grasp", lambda m: bool(m["grasp"])),
        ("rim", lambda m: bool(m["rim"]) and bool(m["grasp"])),
        ("insertion", lambda m: bool(m["insertion"])),
        ("task", lambda m: bool(m["task"])),
    ),
    "raw": tuple((name, (lambda n: lambda m: bool(m[n]))(name)) for name in STAGES),
}


CATEGORY = {"grasp": "grasp", "rim": "insertion", "insertion": "insertion", "task": "task"}


STATE_LADDERS = {
    "square_d1":    {"grasp_m": 0.075, "lift_z": 0.930, "align_xy": 0.10},
    "threading_d0": {"grasp_m": 0.075, "lift_z": 0.950, "align_xy": 0.10},
}
_EEF = slice(0, 3)
_OBJ = slice(9, 12)
_GOAL = slice(23, 26)


def stage_from_state(raw_state, task: str, rewarded: bool = False) -> int:
    """Progress rung of one RAW (unscaled) state vector:"""
    import numpy as np

    if task not in STATE_LADDERS:
        raise ValueError(
            f"no geometric ladder for {task!r}; choose from {sorted(STATE_LADDERS)}. Add one only "
            "with thresholds validated against that task's replay, not by copying another task's."
        )
    t = STATE_LADDERS[task]
    s = np.asarray(raw_state, dtype=np.float64)
    grasped = float(np.linalg.norm(s[_EEF] - s[_OBJ])) < t["grasp_m"]
    lifted = float(s[_OBJ][2]) > t["lift_z"]
    aligned = lifted and float(np.linalg.norm(s[_OBJ][:2] - s[_GOAL][:2])) < t["align_xy"]
    if rewarded:
        return 4
    if aligned:
        return 3


    if lifted and grasped:
        return 2
    return 1 if grasped else 0


class StateAnnotator:
    """OnlineAnnotator's interface, driven by the raw state vector instead of the simulator."""

    def __init__(self, task: str) -> None:
        if task not in STATE_LADDERS:
            raise ValueError(
                f"--critical-ladder geometric has no ladder for {task!r}; "
                f"choose from {sorted(STATE_LADDERS)}"
            )
        self.task = task
        self.stages: list[int] = []

    def reset(self) -> None:
        self.stages = []

    def observe(self, raw_state, rewarded: bool = False) -> int:
        stage = stage_from_state(raw_state, self.task, rewarded)
        self.stages.append(stage)
        return stage

    def finish(self, trajectory_id: int, success: bool):
        return annotate_stage_sequence(self.stages, trajectory_id, action_steps=1, success=success)


def stage_from_metrics(metrics: dict, ladder: str = "gated") -> int:
    """Highest satisfied rung, 1-indexed over STAGES; 0 when none is satisfied."""
    stage = 0
    for index, (_, predicate) in enumerate(LADDERS[ladder], start=1):
        if predicate(metrics):
            stage = index
    return stage


def bitmask_from_metrics(metrics: dict) -> int:
    """All four signals packed into one int, bit i = STAGES[i]."""
    mask = 0
    for index, name in enumerate(STAGES):
        if bool(metrics.get(name, False)):
            mask |= 1 << index
    return mask


@dataclass
class Annotation:
    """One failed trajectory's catastrophe, in BOTH frame and chunk coordinates."""

    trajectory_id: int
    n_frames: int
    action_steps: int
    peak_stage: int
    catastrophic_frame: int | None
    critical_type: str | None
    stages: list[int] = field(default_factory=list, repr=False)

    @property
    def catastrophic_chunk(self) -> int | None:
        if self.catastrophic_frame is None:
            return None
        return self.catastrophic_frame // self.action_steps

    @property
    def n_chunks(self) -> int:


        return -(-self.n_frames // self.action_steps)

    @property
    def frames_after_catastrophe(self) -> int:
        if self.catastrophic_frame is None:
            return 0
        return self.n_frames - 1 - self.catastrophic_frame

    @property
    def suffix_fraction(self) -> float:
        return self.frames_after_catastrophe / self.n_frames if self.n_frames else 0.0

    def critical_chunks(self, pre_window: int) -> set[int]:
        """{t - pre_window, ..., t} in CHUNK coordinates, clipped at the episode start."""
        last = self.catastrophic_chunk
        if last is None:
            return set()
        return set(range(max(0, last - pre_window), last + 1))

    def retained_chunks(self, truncate: bool) -> set[int]:
        """Chunk indices kept in replay."""
        if not truncate or self.catastrophic_chunk is None:
            return set(range(self.n_chunks))
        return set(range(self.catastrophic_chunk + 1))


def annotate_stage_sequence(
    stages: list[int], trajectory_id: int, action_steps: int, success: bool = False
) -> Annotation:
    """Reduce a per-frame stage sequence to one catastrophe."""
    n_frames = len(stages)
    peak = max(stages) if stages else 0
    if success or peak == 0 or n_frames == 0:


        return Annotation(trajectory_id, n_frames, action_steps, peak, None, None, stages)

    last_at_peak = max(index for index, stage in enumerate(stages) if stage == peak)
    if last_at_peak >= n_frames - 1:

        return Annotation(trajectory_id, n_frames, action_steps, peak, None, None, stages)

    return Annotation(
        trajectory_id=trajectory_id,
        n_frames=n_frames,
        action_steps=action_steps,
        peak_stage=peak,
        catastrophic_frame=last_at_peak,
        critical_type=STAGES[peak - 1],
        stages=stages,
    )


TYPE_CODES: dict[str | None, int] = {None: 0, **{name: i for i, name in enumerate(STAGES, start=1)}}
CODE_TYPES: dict[int, str | None] = {code: name for name, code in TYPE_CODES.items()}


def task_metrics(env) -> dict:
    """MimicGen's per-timestep signals from whatever wrapper depth the env is behind."""
    while not hasattr(env, "_get_partial_task_metrics"):
        inner = getattr(env, "_env", None)
        if inner is None:
            raise AttributeError(
                "no _get_partial_task_metrics() below this env -- critical-state annotation is "
                "only defined for MimicGen tasks that expose partial task metrics"
            )
        env = inner
    return env._get_partial_task_metrics()


class OnlineAnnotator:
    """Per-chunk stage tracker for a live rollout, annotated the moment the episode ends."""

    def __init__(self, ladder: str = "gated") -> None:
        if ladder not in LADDERS:
            raise ValueError(f"unknown ladder {ladder!r}; choose from {sorted(LADDERS)}")
        self.ladder = ladder
        self.stages: list[int] = []

    def reset(self) -> None:
        self.stages = []

    def observe(self, env) -> int:
        """Record the stage of the state the chunk just landed in."""
        stage = stage_from_metrics(task_metrics(env), self.ladder)
        self.stages.append(stage)
        return stage

    def finish(self, trajectory_id: int, success: bool) -> Annotation:
        return annotate_stage_sequence(self.stages, trajectory_id, action_steps=1, success=success)


def render_timeline(annotation: Annotation, pre_window: int, width: int = 1) -> list[str]:
    """The compact per-chunk trace the prompt's annotation_validation section asks for."""
    critical = annotation.critical_chunks(pre_window)
    retained = annotation.retained_chunks(truncate=True)
    catastrophic = annotation.catastrophic_chunk
    lines = []
    for chunk in range(annotation.n_chunks):
        if chunk == catastrophic:
            label = f"CATASTROPHIC ({annotation.critical_type})"
        elif chunk not in retained:
            label = "removed"
        elif chunk in critical:
            label = "critical"
        else:
            label = "normal"
        lines.append(f"{chunk:03d} {label}")
    return lines

"""The campaign matrix: every TASK x ARM the reference launchers define, and how each maps onto configs/.

One Cell = one (config file, task, variant). `live_arm` is the ARM= value of the reference launcher that produced
the golden file (reference/launchers/); `campaign` says whether the reference campaign actually ran the cell
(3 seeds each: 43 and 44 at 600k env steps, 45 at 1M). The four DSRL cells with campaign=False are defined by the
launcher and resolved here, but have no reference result.
"""
from __future__ import annotations

from dataclasses import dataclass

TASKS = ("square_d0", "square_d1", "threading_d0", "hammer_cleanup_d1")
GOLDEN_SEED = 99                       # a seed no reference run uses: goldens can never be mistaken for a run
GOLDEN_STEPS = (600_000, 1_000_000)    # the two budgets of the campaign; the golden config is the 600k one


@dataclass(frozen=True)
class Cell:
    method: str          # dice_rl | dsrl
    config: str          # configs/<method>/<config>.sh
    task: str
    variant: str         # dice_rl dvd only: tdonly | tddistill (the DISTILL= parameter); "" elsewhere
    live_arm: str        # ARM= of the reference launcher
    campaign: bool = True

    @property
    def id(self) -> str:
        return f"{self.config}{'_' + self.variant if self.variant else ''}__{self.task}"

    @property
    def launcher(self) -> str:
        return "launch_dicerl_ab.sh" if self.method == "dice_rl" else "launch_hammer_dsrl.sh"

    @property
    def config_path(self) -> str:
        return f"configs/{self.method}/{self.config}.sh"

    @property
    def golden(self) -> str:
        return f"golden/{self.method}/{self.id}.json"


def default_variant(task: str) -> str:
    """configs/dice_rl/dvd.sh without DISTILL=: the paper recipe (square_d0 TD + distill, the other tasks TD only)."""
    return "tddistill" if task == "square_d0" else "tdonly"


def _dice_rl() -> list[Cell]:
    cells = []
    for task in TASKS:
        for variant in ("tdonly", "tddistill"):
            if variant == default_variant(task):
                arm = "dvd"                                        # the paper recipe of the task
            else:
                arm = "dvdtd" if task == "square_d0" else "dvdd"   # the distill A/B twin
            cells.append(Cell("dice_rl", "dvd", task, variant, arm))
        cells.append(Cell("dice_rl", "q", task, "", "q"))
    return cells


# DSRL arms the reference campaign ran, per task (queue of 48 runs = 16 cells x 3 seeds)
_DSRL_RAN = {
    "ours": TASKS,
    "q": TASKS,
    "oursdist": ("square_d0", "hammer_cleanup_d1"),
    "oursmc": ("square_d0", "square_d1", "hammer_cleanup_d1"),
    "randv": ("square_d0", "square_d1", "hammer_cleanup_d1"),
}


def _dsrl() -> list[Cell]:
    return [Cell("dsrl", arm, task, "", arm, task in _DSRL_RAN[arm])
            for arm in ("ours", "oursmc", "oursdist", "randv", "q") for task in TASKS]


CELLS: tuple[Cell, ...] = tuple(_dice_rl() + _dsrl())

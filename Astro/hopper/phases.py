"""Phase container: decision variables of one trajectory segment plus its constraint list."""
from __future__ import annotations

from dataclasses import dataclass, field

import casadi as ca


@dataclass
class Phase:
    name: str
    kind: str                 # rise | ascent | coast | braking | observation | terminal
    n: int                    # collocation intervals (n + 1 nodes)
    powered: bool
    X: object                 # 7 x (n+1), nondimensional (offset position r, v, m)
    U: object                 # 4 x (n+1), nondimensional [T_vec, Gamma]; zeros if coast
    tdur: object              # nondimensional duration (variable or fixed number)
    branch: str | None = None
    site: object = None       # nondimensional offset position of this branch's landing site (DM 3x1)
    site_up: object = None    # local vertical at that site (DM 3x1)
    prev: "Phase | None" = None      # preceding phase in the chain (for junction constraints)
    constraints: list = field(default_factory=list)
    t_bounds: tuple | None = None    # nondimensional (lo, hi) if duration is free
    launch_up: object = None         # local vertical at actual launch point; may be symbolic on hop 2
    landing_target: object = None    # nominal center of the permitted landing region
    sensor_sites: list = field(default_factory=list)  # observed point offsets from mission start
    stage: int = 0                   # burn stage index within its kind (ascent and braking may have several)
    mode: object = None              # EngineMode of this stage; None for coast
    hop: int = 1

    def apply_constraints(self, opti):
        for c in self.constraints:
            c.apply(opti, self)


def stage_nodes(total: int, n_stages: int) -> int:
    """`solver.nodes[kind]` is the total for the kind; a multi-stage burn shares it between its stages."""
    return total if n_stages == 1 else max(-(-total // n_stages), 6)


def stage_name(base: str, stage: int) -> str:
    """The first stage keeps the plain phase name; later stages get a suffix (ascent -> ascent_s2)."""
    return base if stage == 0 else f"{base}_s{stage + 1}"


def make_phase(opti, name, kind, n, powered, *, t_bounds=None, t_fixed=None, branch=None,
               site=None, site_up=None, prev=None, launch_up=None, stage=0, mode=None, hop=1) -> Phase:
    X = opti.variable(7, n + 1)
    U = opti.variable(4, n + 1) if powered else ca.DM.zeros(4, n + 1)
    if t_fixed is not None:
        tdur = float(t_fixed)
    else:
        tdur = opti.variable()
        opti.subject_to(opti.bounded(t_bounds[0], tdur, t_bounds[1]))
    return Phase(name, kind, n, powered, X, U, tdur, branch, site, site_up, prev, [], t_bounds, launch_up,
                 stage=stage, mode=mode, hop=hop)

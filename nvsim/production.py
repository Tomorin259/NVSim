"""Master-regulator forcing definitions for NVSim.

This module collects the two supported ways to supply external production for
source/master regulators:

- state/bin-wise production tables via ``StateProductionProfile``;
- time-dependent scalar programs via ``AlphaProgram`` and helper constructors.

Keeping both mechanisms in one file makes the master-regulator forcing layer
readable without splitting closely related code across multiple tiny modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import warnings

import numpy as np
import pandas as pd


PROGRAM_KINDS = {
    "constant",
    "linear_increase",
    "linear_decrease",
    "sigmoid_increase",
    "sigmoid_decrease",
}

ALPHA_SOURCE_MODES = {"continuous_program", "state_anchor"}
TRANSITION_SCHEDULES = {"step", "linear", "sigmoid"}


def transition_weight(
    fraction: float,
    schedule: str = "sigmoid",
    midpoint: float = 0.5,
    steepness: float = 10.0,
) -> float:
    """Return interpolation weight for state-anchor master alpha transitions."""

    if schedule not in TRANSITION_SCHEDULES:
        raise ValueError("transition_schedule must be 'step', 'linear', or 'sigmoid'")
    if midpoint < 0 or midpoint > 1:
        raise ValueError("transition_midpoint must be in [0, 1]")
    if steepness <= 0:
        raise ValueError("transition_steepness must be positive")
    x = float(np.clip(fraction, 0.0, 1.0))
    if schedule == "step":
        # Step transitions switch to the child anchor at the segment boundary.
        # Unlike linear/sigmoid schedules, there is no in-segment interpolation.
        return 1.0
    if schedule == "linear":
        return x

    lo = 1.0 / (1.0 + np.exp(-steepness * (0.0 - midpoint)))
    hi = 1.0 / (1.0 + np.exp(-steepness * (1.0 - midpoint)))
    value = 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))
    if np.isclose(hi, lo):
        return x
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


@dataclass(frozen=True)
class AlphaProgram:
    """Time-dependent alpha program for one master regulator.

    ``t`` is normalized pseudotime in [0, 1]. The output is always clipped to a
    non-negative value.
    """

    kind: str = "constant"
    start: float = 1.0
    end: float = 1.0
    midpoint: float = 0.5
    steepness: float = 10.0

    def __post_init__(self) -> None:
        if self.kind not in PROGRAM_KINDS:
            raise ValueError(f"unknown alpha program kind {self.kind!r}")
        if self.start < 0 or self.end < 0:
            raise ValueError("alpha program start/end must be non-negative")
        if self.steepness <= 0:
            raise ValueError("alpha program steepness must be positive")

    def value(self, t: float) -> float:
        t = float(np.clip(t, 0.0, 1.0))
        if self.kind == "constant":
            value = self.start
        elif self.kind == "linear_increase":
            value = self.start + (self.end - self.start) * t
        elif self.kind == "linear_decrease":
            value = self.end + (self.start - self.end) * (1.0 - t)
        elif self.kind == "sigmoid_increase":
            z = 1.0 / (1.0 + np.exp(-self.steepness * (t - self.midpoint)))
            value = self.start + (self.end - self.start) * z
        elif self.kind == "sigmoid_decrease":
            z = 1.0 / (1.0 + np.exp(-self.steepness * (t - self.midpoint)))
            value = self.end + (self.start - self.end) * (1.0 - z)
        else:
            raise ValueError(f"unknown alpha program kind {self.kind!r}")
        return float(max(value, 0.0))


def constant(value: float) -> AlphaProgram:
    return AlphaProgram(kind="constant", start=value, end=value)


def linear_increase(start: float, end: float) -> AlphaProgram:
    return AlphaProgram(kind="linear_increase", start=start, end=end)


def linear_decrease(start: float, end: float) -> AlphaProgram:
    return AlphaProgram(kind="linear_decrease", start=start, end=end)


def sigmoid_increase(start: float, end: float, midpoint: float = 0.5, steepness: float = 10.0) -> AlphaProgram:
    return AlphaProgram(kind="sigmoid_increase", start=start, end=end, midpoint=midpoint, steepness=steepness)


def sigmoid_decrease(start: float, end: float, midpoint: float = 0.5, steepness: float = 10.0) -> AlphaProgram:
    return AlphaProgram(kind="sigmoid_decrease", start=start, end=end, midpoint=midpoint, steepness=steepness)


def coerce_programs(programs: Mapping[str, AlphaProgram | float] | None) -> dict[str, AlphaProgram]:
    """Normalize user-provided master-regulator programs."""

    if programs is None:
        return {}
    coerced: dict[str, AlphaProgram] = {}
    for gene, program in programs.items():
        if isinstance(program, AlphaProgram):
            coerced[str(gene)] = program
        else:
            coerced[str(gene)] = constant(float(program))
    return coerced


def evaluate_programs(
    programs: Mapping[str, AlphaProgram | float],
    genes: list[str] | tuple[str, ...],
    t: float,
    default: float = 0.5,
) -> pd.Series:
    """Evaluate master-regulator programs on a fixed gene order."""

    coerced = coerce_programs(programs)
    values = []
    for gene in genes:
        program = coerced.get(str(gene), constant(default))
        values.append(program.value(t))
    return pd.Series(values, index=pd.Index([str(g) for g in genes], name="gene"), dtype=float)


@dataclass(frozen=True)
class StateProductionProfile:
    """Master/source production rates indexed by discrete state."""

    rates: pd.DataFrame

    def __post_init__(self) -> None:
        rates = self.rates.copy()
        if rates.empty:
            raise ValueError("production rates must not be empty")
        rates.index = rates.index.astype(str)
        rates.columns = rates.columns.astype(str)
        rates = rates.apply(pd.to_numeric, errors="raise")
        values = rates.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("production rates must be finite")
        if (values < 0).any():
            raise ValueError("production rates must be non-negative")
        object.__setattr__(self, "rates", rates.astype(float))

    @property
    def states(self) -> tuple[str, ...]:
        return tuple(str(state) for state in self.rates.index)

    @property
    def genes(self) -> tuple[str, ...]:
        return tuple(str(gene) for gene in self.rates.columns)

    def source_alpha(self, state: str, genes: list[str] | tuple[str, ...] | pd.Index | None = None) -> pd.Series:
        """Return source alpha values for one state."""

        state = str(state)
        if state not in self.rates.index:
            raise ValueError(f"unknown production state {state!r}")
        alpha = self.rates.loc[state].copy()
        if genes is not None:
            alpha = alpha.reindex([str(gene) for gene in genes], fill_value=0.0)
        alpha.index.name = "gene"
        return alpha.astype(float)

    def source_alpha_transition(
        self,
        parent_state: str,
        child_state: str,
        fraction: float,
        schedule: str = "sigmoid",
        midpoint: float = 0.5,
        steepness: float = 10.0,
        genes: list[str] | tuple[str, ...] | pd.Index | None = None,
    ) -> pd.Series:
        """Interpolate master alpha from a parent state anchor to a child anchor."""

        if fraction < 0 or fraction > 1:
            raise ValueError("fraction must be in [0, 1]")
        parent = self.source_alpha(parent_state)
        child = self.source_alpha(child_state)
        weight = transition_weight(fraction, schedule=schedule, midpoint=midpoint, steepness=steepness)
        alpha = (1.0 - weight) * parent + weight * child
        if genes is not None:
            alpha = alpha.reindex([str(gene) for gene in genes], fill_value=0.0)
        alpha.index.name = "gene"
        return alpha.astype(float)

    def validate_master_genes(self, master_genes: list[str] | tuple[str, ...] | pd.Index) -> None:
        """Ensure the production profile has exactly the expected master genes."""

        expected = {str(gene) for gene in master_genes}
        observed = set(self.genes)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ValueError("production profile genes do not match master genes: " + ", ".join(details))

    def validate_states(self, states: list[str] | tuple[str, ...] | pd.Index) -> None:
        missing = [str(state) for state in states if str(state) not in self.rates.index]
        if missing:
            raise ValueError(f"unknown production state(s): {missing}")

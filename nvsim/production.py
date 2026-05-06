"""State-based production profiles for source/master regulator genes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StateProductionProfile:
    """Master/source production rates indexed by discrete state.

    ``rates`` is a states x genes table. Rows are states such as ``bin_0`` or
    ``branch_0``. Columns are source/master regulator gene ids. Values are
    non-negative production rates that can be used as alpha values for those
    source genes.
    """

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
        """Return source alpha values for one state.

        If ``genes`` is provided, the returned Series is reindexed to that order
        and missing genes are filled with zero. This is convenient for expanding
        a master-only profile onto a full GRN gene order.
        """

        state = str(state)
        if state not in self.rates.index:
            raise ValueError(f"unknown production state {state!r}")
        alpha = self.rates.loc[state].copy()
        if genes is not None:
            alpha = alpha.reindex([str(gene) for gene in genes], fill_value=0.0)
        alpha.index.name = "gene"
        return alpha.astype(float)

    def source_alpha_interpolated(
        self,
        parent_state: str,
        child_state: str,
        fraction: float,
        genes: list[str] | tuple[str, ...] | pd.Index | None = None,
    ) -> pd.Series:
        """Linearly interpolate source alpha between two states."""

        if fraction < 0 or fraction > 1:
            raise ValueError("fraction must be in [0, 1]")
        parent = self.source_alpha(parent_state)
        child = self.source_alpha(child_state)
        alpha = parent + float(fraction) * (child - parent)
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

"""High-level simulation mode helpers and graph schemas."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DifferentiationGraph:
    """A rooted DAG over discrete states/cell types.

    The canonical edge schema is:
    - ``parent_state``
    - ``child_state``

    Optional extra columns are preserved as metadata but do not affect
    validation or traversal in the current implementation.
    """

    edges: pd.DataFrame
    states: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        edges = self.edges.copy()
        required = {"parent_state", "child_state"}
        missing = required - set(edges.columns)
        if missing:
            raise ValueError(f"differentiation graph is missing columns: {sorted(missing)}")
        edges["parent_state"] = edges["parent_state"].astype(str)
        edges["child_state"] = edges["child_state"].astype(str)
        if edges[["parent_state", "child_state"]].isna().any().any():
            raise ValueError("differentiation graph states must not be null")
        if edges.empty and self.states is None:
            raise ValueError("differentiation graph must define at least one edge or explicit states")
        if edges.duplicated(subset=["parent_state", "child_state"]).any():
            raise ValueError("differentiation graph contains duplicate parent->child edges")

        explicit_states = tuple(str(state) for state in self.states) if self.states is not None else ()
        all_states = tuple(
            dict.fromkeys(
                [
                    *explicit_states,
                    *edges["parent_state"].tolist(),
                    *edges["child_state"].tolist(),
                ]
            )
        )
        if not all_states:
            raise ValueError("differentiation graph must contain at least one state")

        parent_counts = edges.groupby("child_state")["parent_state"].nunique()
        multi_parent = sorted(parent_counts[parent_counts > 1].index.astype(str).tolist())
        if multi_parent:
            raise ValueError(
                "each child state must have at most one parent in the differentiation graph: "
                f"{multi_parent}"
            )

        object.__setattr__(self, "edges", edges.reset_index(drop=True))
        object.__setattr__(self, "states", all_states)

        # Trigger validation early so invalid graphs fail at construction time.
        self.topological_order()

    @property
    def root_states(self) -> tuple[str, ...]:
        child_states = set(self.edges["child_state"].astype(str))
        return tuple(state for state in self.states or () if state not in child_states)

    def parent_of(self, state: str) -> str | None:
        state = str(state)
        parents = self.edges.loc[self.edges["child_state"] == state, "parent_state"].astype(str).unique().tolist()
        if not parents:
            return None
        if len(parents) > 1:
            raise ValueError(f"state {state!r} has multiple parents, which is unsupported")
        return parents[0]

    def children_of(self, state: str) -> tuple[str, ...]:
        state = str(state)
        children = self.edges.loc[self.edges["parent_state"] == state, "child_state"].astype(str).tolist()
        return tuple(children)

    def state_depths(self) -> dict[str, int]:
        order = self.topological_order()
        depths: dict[str, int] = {}
        for state in order:
            parent = self.parent_of(state)
            depths[state] = 0 if parent is None else depths[parent] + 1
        return depths

    def topological_order(self) -> tuple[str, ...]:
        outgoing: dict[str, list[str]] = {state: [] for state in self.states or ()}
        indegree: dict[str, int] = {state: 0 for state in self.states or ()}
        for edge in self.edges.itertuples(index=False):
            parent = str(edge.parent_state)
            child = str(edge.child_state)
            outgoing[parent].append(child)
            indegree[child] += 1

        queue = [state for state in self.states or () if indegree[state] == 0]
        order: list[str] = []
        while queue:
            state = queue.pop(0)
            order.append(state)
            for child in outgoing[state]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.states or ()):
            raise ValueError("differentiation graph must be acyclic")
        return tuple(order)

    def validate_states(self, states: list[str] | tuple[str, ...] | pd.Index) -> None:
        observed = set(self.states or ())
        missing = [str(state) for state in states if str(state) not in observed]
        if missing:
            raise ValueError(f"unknown differentiation graph state(s): {missing}")


def coerce_differentiation_graph(
    graph: DifferentiationGraph | pd.DataFrame | dict[str, object] | None,
) -> DifferentiationGraph | None:
    if graph is None or isinstance(graph, DifferentiationGraph):
        return graph
    if isinstance(graph, pd.DataFrame):
        return DifferentiationGraph(graph)
    if isinstance(graph, dict):
        if "edges" not in graph:
            raise ValueError("differentiation graph dict input must contain an 'edges' entry")
        edges = graph["edges"]
        if not isinstance(edges, pd.DataFrame):
            edges = pd.DataFrame(edges)
        states = graph.get("states")
        return DifferentiationGraph(edges=edges, states=None if states is None else tuple(str(state) for state in states))
    raise TypeError("differentiation_graph must be a DifferentiationGraph, DataFrame, dict, or None")

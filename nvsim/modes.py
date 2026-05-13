"""Graph schemas and topology helpers for the unified NVSim simulator."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StateGraph:
    """A rooted DAG over discrete simulation states.

    Required edge columns:
    - ``parent_state``
    - ``child_state``

    Extra columns are preserved as metadata but do not affect validation or
    traversal in the current implementation.
    """

    edges: pd.DataFrame
    states: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        edges = self.edges.copy()
        required = {"parent_state", "child_state"}
        missing = required - set(edges.columns)
        if missing:
            raise ValueError(f"state graph is missing columns: {sorted(missing)}")
        edges["parent_state"] = edges["parent_state"].astype(str)
        edges["child_state"] = edges["child_state"].astype(str)
        if edges.empty and self.states is None:
            raise ValueError("state graph must define at least one edge or explicit states")
        if edges.duplicated(subset=["parent_state", "child_state"]).any():
            raise ValueError("state graph contains duplicate parent->child edges")

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
            raise ValueError("state graph must contain at least one state")

        parent_counts = edges.groupby("child_state")["parent_state"].nunique()
        multi_parent = sorted(parent_counts[parent_counts > 1].index.astype(str).tolist())
        if multi_parent:
            raise ValueError(
                "each child state must have at most one parent in the state graph: "
                f"{multi_parent}"
            )

        object.__setattr__(self, "edges", edges.reset_index(drop=True))
        object.__setattr__(self, "states", all_states)
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
            raise ValueError("state graph must be acyclic")
        return tuple(order)

    def validate_states(self, states: list[str] | tuple[str, ...] | pd.Index) -> None:
        observed = set(self.states or ())
        missing = [str(state) for state in states if str(state) not in observed]
        if missing:
            raise ValueError(f"unknown state graph state(s): {missing}")


DifferentiationGraph = StateGraph


def coerce_graph(
    graph: StateGraph | pd.DataFrame | dict[str, object] | None,
) -> StateGraph | None:
    if graph is None or isinstance(graph, StateGraph):
        return graph
    if isinstance(graph, pd.DataFrame):
        return StateGraph(graph)
    if isinstance(graph, dict):
        if "edges" not in graph:
            raise ValueError("graph dict input must contain an edges entry")
        edges = graph["edges"]
        if not isinstance(edges, pd.DataFrame):
            edges = pd.DataFrame(edges)
        states = graph.get("states")
        return StateGraph(edges=edges, states=None if states is None else tuple(str(state) for state in states))
    raise TypeError("graph must be a StateGraph, DataFrame, dict, or None")


def coerce_differentiation_graph(
    graph: StateGraph | pd.DataFrame | dict[str, object] | None,
) -> StateGraph | None:
    return coerce_graph(graph)


def path_graph(states: list[str] | tuple[str, ...]) -> StateGraph:
    ordered = [str(state) for state in states]
    if len(ordered) < 1:
        raise ValueError("path_graph requires at least one state")
    if len(ordered) == 1:
        return StateGraph(pd.DataFrame(columns=["parent_state", "child_state"]), states=tuple(ordered))
    return StateGraph(
        pd.DataFrame(
            {
                "parent_state": ordered[:-1],
                "child_state": ordered[1:],
            }
        ),
        states=tuple(ordered),
    )


def branching_graph(root_state: str, child_states: list[str] | tuple[str, ...]) -> StateGraph:
    children = [str(state) for state in child_states]
    if not children:
        raise ValueError("branching_graph requires at least one child state")
    return StateGraph(
        pd.DataFrame(
            {
                "parent_state": [str(root_state)] * len(children),
                "child_state": children,
            }
        ),
        states=(str(root_state), *children),
    )

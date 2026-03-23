"""
Part 1: Network Representation
================================
Directed graph with nodes, links, travel costs, and OD demands.
Supports fixed and flow-dependent (BPR-style) link costs.
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class Link:
    """A directed network link."""
    id: int
    u: int          # tail node
    v: int          # head node
    cost: float     # current (possibly flow-dependent) cost
    free_flow: float
    capacity: float = float('inf')
    flow: float = 0.0

    def bpr_cost(self, flow: Optional[float] = None) -> float:
        """BPR-style travel cost: t0 * (1 + 0.15*(f/C)^4)"""
        f = flow if flow is not None else self.flow
        if self.capacity == float('inf'):
            return self.free_flow
        return self.free_flow * (1 + 0.15 * (f / self.capacity) ** 4)

    def quadratic_cost(self, flow: Optional[float] = None) -> float:
        """Paper's congestion model: t0 + (f/C)^2"""
        f = flow if flow is not None else self.flow
        if self.capacity == float('inf'):
            return self.free_flow
        return self.free_flow + (f / self.capacity) ** 2


@dataclass
class ODPair:
    """An Origin-Destination movement with demand."""
    origin: int
    destination: int
    demand: float


class Network:
    """
    Directed road network.
    
    Stores links, nodes, OD pairs, and provides shortest-path utilities.
    """

    def __init__(self, name: str = "Network"):
        self.name = name
        self.graph = nx.DiGraph()
        self.links: Dict[int, Link] = {}           # link_id -> Link
        self.link_lookup: Dict[Tuple, int] = {}    # (u,v) -> link_id
        self.od_pairs: List[ODPair] = []
        self._sp_cache: Dict = {}                  # memoised shortest paths

    def add_node(self, node_id: int, **attrs):
        self.graph.add_node(node_id, **attrs)

    def add_link(self, link_id: int, u: int, v: int,
                 free_flow: float, capacity: float = float('inf')):
        """Add a directed link with given free-flow cost and capacity."""
        link = Link(id=link_id, u=u, v=v,
                    cost=free_flow, free_flow=free_flow, capacity=capacity)
        self.links[link_id] = link
        self.link_lookup[(u, v)] = link_id
        self.graph.add_edge(u, v, link_id=link_id, weight=free_flow)

    def add_od(self, origin: int, destination: int, demand: float):
        self.od_pairs.append(ODPair(origin, destination, demand))

    def update_link_costs(self, cost_fn: str = 'fixed'):
        """
        Recompute link costs based on current flows.
        cost_fn: 'fixed' | 'bpr' | 'quadratic'
        """
        for link_id, link in self.links.items():
            if cost_fn == 'bpr':
                link.cost = link.bpr_cost()
            elif cost_fn == 'quadratic':
                link.cost = link.quadratic_cost()
            else:
                link.cost = link.free_flow
            # Update graph edge weight for shortest path calcs
            self.graph[link.u][link.v]['weight'] = link.cost

    def get_link_cost(self, u: int, v: int) -> float:
        lid = self.link_lookup.get((u, v))
        if lid is None:
            return float('inf')
        return self.links[lid].cost

    # ------------------------------------------------------------------
    # All-pairs shortest paths (memoised for branch-and-bound efficiency)
    # ------------------------------------------------------------------

    def compute_all_pairs_shortest_paths(self):
        """
        Pre-compute shortest cost distances between all node pairs.
        Stored in self._sp_cache for fast O(1) lookup in branch-and-bound.
        """
        self._sp_cache = dict(
            nx.all_pairs_dijkstra_path_length(self.graph, weight='weight')
        )
        return self._sp_cache

    def shortest_path_cost(self, src: int, dst: int) -> float:
        """Return shortest-path cost from src to dst (uses cache if available)."""
        if self._sp_cache:
            return self._sp_cache.get(src, {}).get(dst, float('inf'))
        try:
            return nx.dijkstra_path_length(self.graph, src, dst, weight='weight')
        except nx.NetworkXNoPath:
            return float('inf')

    def shortest_path(self, src: int, dst: int) -> List[int]:
        """Return the shortest path (list of nodes) from src to dst."""
        try:
            return nx.dijkstra_path(self.graph, src, dst, weight='weight')
        except nx.NetworkXNoPath:
            return []

    def route_cost(self, route: List[int]) -> float:
        """Total cost of a route (list of nodes)."""
        total = 0.0
        for i in range(len(route) - 1):
            total += self.get_link_cost(route[i], route[i + 1])
        return total

    def nodes(self):
        return list(self.graph.nodes())

    def __repr__(self):
        return (f"Network('{self.name}', nodes={self.graph.number_of_nodes()}, "
                f"links={self.graph.number_of_edges()}, ODs={len(self.od_pairs)})")


# ===========================================================================
# Factory functions: build the toy networks from the paper
# ===========================================================================

def build_fig4_network() -> Network:
    """
    Fig. 4 toy network: O -> D with 3 routes.
    Route 1 (top): O->(node1)->(node2)->D  costs: 100, 10, 100  total=210
    Route 2 (mid): O->(node1)->D            costs: 100, 100      total=200
    Route 3 (bot): O->D                     costs: 201            total=201
    Scenario A (as in paper top table)
    """
    net = Network("Fig4_ScenarioA")
    for n in [0, 1, 2, 3]:  # 0=O, 3=D
        net.add_node(n)
    # Route 1: O->1->2->D  (local detour via top)
    net.add_link(1, 0, 1, free_flow=100)
    net.add_link(2, 1, 2, free_flow=10)
    net.add_link(3, 2, 3, free_flow=100)
    # Route 2: O->1->D  (direct mid)
    net.add_link(4, 1, 3, free_flow=100)
    # Route 3: O->D  (direct bottom)
    net.add_link(5, 0, 3, free_flow=201)
    net.add_od(0, 3, demand=1.0)
    return net


def build_fig6_network() -> Network:
    """
    Fig. 6 example network: A->E for demonstrating local detouredness.
    Nodes: A=0, B=1, C=2, D=3, E=4
    Links: A-B=10, B-C=5, B-D=10, A-D=25, C-D=0 (implied), D-E=5
    Routes:
      Route 1: A->B->C->D->E  (goes via C, local detour at B->D segment)
      Route 2: A->B->D->E     (direct, shortest)
      Route 3: A->D->E        (goes via A-D directly)
    """
    net = Network("Fig6_LocalDetour")
    for n in range(5):  # A=0,B=1,C=2,D=3,E=4
        net.add_node(n)
    net.add_link(1, 0, 1, free_flow=10)   # A->B
    net.add_link(2, 1, 2, free_flow=10)   # B->C (cost=10, so B->C->D=15)
    net.add_link(3, 2, 3, free_flow=5)    # C->D (cost=5, B->C->D=15 vs B->D=10 => detour 0.5)
    net.add_link(4, 1, 3, free_flow=10)   # B->D  (direct)
    net.add_link(5, 0, 3, free_flow=25)   # A->D  (direct, cost=25)
    net.add_link(6, 3, 4, free_flow=5)    # D->E
    net.add_od(0, 4, demand=1.0)
    return net


def build_fig7_network(x: float = 10.0) -> Network:
    """
    Fig. 7 small-scale network: O->D with 3 routes, parameterised by x.
    Link 1: O->mid  cost=x     (shared by Routes 1 & 2)
    Link 2: mid->D  cost=25-x  (Route 1 upper branch)
    Link 3: mid->D  cost=20-x  (Route 2 lower branch)
    Link 4: O->D    cost=35    (Route 3 direct)

    Route 1: O->mid->D via link2, cost=x+(25-x)=25
    Route 2: O->mid->D via link3, cost=x+(20-x)=20  (cheapest)
    Route 3: O->D via link4,      cost=35
    """
    net = Network(f"Fig7_x={x}")
    # Nodes: 0=O, 1=mid-upper, 2=mid-lower-junction, 3=D
    # To have distinct routes sharing a link, use: O->hub, hub->D (upper), hub->D (lower)
    # NetworkX only allows one edge per (u,v) pair, so we use an intermediate node
    for n in [0, 1, 2, 3]:
        net.add_node(n)
    net.add_link(1, 0, 1, free_flow=x)       # Link 1: O -> hub
    net.add_link(2, 1, 3, free_flow=25 - x)  # Link 2: hub -> D (upper, Route 1)
    # For Route 2 we need a separate path; use node 2 as intermediate
    net.add_link(3, 0, 2, free_flow=x)        # Link 3a: O -> hub2
    net.add_link(4, 2, 3, free_flow=20 - x)   # Link 3b: hub2 -> D (lower, Route 2)
    net.add_link(5, 0, 3, free_flow=35)        # Link 4: O -> D direct (Route 3)
    net.add_od(0, 3, demand=1.0)
    return net


def build_fig16_network(x: float = 0.0) -> Network:
    """
    Fig. 16 (Section 5.3.1): 4-link congested network.
    Links: (1) O->hub, (2) hub->D_upper, (3) hub->D_lower, (4) O->D_direct
    Free-flow: t0=(50,10,5,50), Capacities: C=(1000,1000,100,1000)
    Route 1: links 1+3  (O->hub->D via lower, detours locally on hub)
    Route 2: links 1+2  (O->hub->D via upper)
    Route 3: link  4    (O->D direct)
    """
    net = Network("Fig16_Congested")
    for n in [0, 1, 2, 3]:  # 0=O, 1=hub, 2=D_upper_branch, 3=D
        net.add_node(n)
    t0 = [50, 10, 5, 50]
    C  = [1000, 1000, 100, 1000]
    # Link 1: O->hub
    net.add_link(1, 0, 1, free_flow=t0[0], capacity=C[0])
    # Link 2: hub->D (upper branch, Route 2 uses this)
    net.add_link(2, 1, 2, free_flow=t0[1], capacity=C[1])
    net.add_link(22, 2, 3, free_flow=0.0,  capacity=float('inf'))  # connector
    # Link 3: hub->D (lower branch, Route 1 uses this — the "detour" path)
    net.add_link(3, 1, 3, free_flow=t0[2], capacity=C[2])
    # Link 4: O->D direct
    net.add_link(4, 0, 3, free_flow=t0[3], capacity=C[3])
    net.add_od(0, 3, demand=5000.0)
    return net

"""
Part 2: Local Detouredness Measure (Equation 15)
==================================================
For a route, computes φ_mi = max over all segments (a,b) of:
    (cost_of_used_sub_route - min_cost_alternative) / min_cost_alternative

Key insight from paper: we only need the *most detouring* segment —
so during branch-and-bound we track this incrementally.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from network import Network


def local_detouredness(route: List[int], net: Network) -> float:
    """
    Compute the local detouredness φ of a route (Eq. 15).

    For every ordered node pair (a, b) where a appears before b in the route,
    compare the cost of the sub-route actually used vs. the cheapest alternative.

    φ = max over all segments { (used_cost - min_cost) / min_cost }

    Parameters
    ----------
    route : list of node IDs in traversal order
    net   : Network with pre-computed all-pairs shortest paths

    Returns
    -------
    float : local detouredness (0 means no detour anywhere)
    """
    if len(route) < 2:
        return 0.0

    max_detour = 0.0

    # Enumerate all ordered node pairs (a, b) with a before b in the route
    for i in range(len(route)):
        for j in range(i + 1, len(route)):
            a, b = route[i], route[j]

            # Cost of the sub-route actually used (traversing route[i..j])
            used_sub_cost = 0.0
            for k in range(i, j):
                used_sub_cost += net.get_link_cost(route[k], route[k + 1])

            # Cheapest alternative between a and b (from pre-computed SP)
            min_cost = net.shortest_path_cost(a, b)

            if min_cost <= 0:
                continue  # avoid division by zero; nodes are same or no path

            relative_detour = (used_sub_cost - min_cost) / min_cost
            # Clamp numerical noise
            relative_detour = max(0.0, relative_detour)

            if relative_detour > max_detour:
                max_detour = relative_detour

    return max_detour


def local_detouredness_incremental(
    partial_route: List[int],
    new_node: int,
    current_max_detour: float,
    net: Network
) -> float:
    """
    Efficiently update local detouredness when extending a partial route
    by one new node. Only checks NEW segments ending at new_node.

    Used inside the branch-and-bound to avoid re-computing all segments.

    Parameters
    ----------
    partial_route      : current route prefix (nodes visited so far)
    new_node           : node being added
    current_max_detour : max detour computed so far for partial_route
    net                : Network

    Returns
    -------
    float : updated max detouredness after adding new_node
    """
    max_detour = current_max_detour
    b = new_node

    # Check segments from each previously visited node a to new_node b
    for i, a in enumerate(partial_route):
        # Cost of sub-route used: partial_route[i..end] + new_node
        used_sub_cost = 0.0
        for k in range(i, len(partial_route) - 1):
            used_sub_cost += net.get_link_cost(partial_route[k], partial_route[k + 1])
        used_sub_cost += net.get_link_cost(partial_route[-1], b)

        # Shortest-path cost from a to b
        min_cost = net.shortest_path_cost(a, b)

        if min_cost <= 0:
            continue

        relative_detour = max(0.0, (used_sub_cost - min_cost) / min_cost)
        if relative_detour > max_detour:
            max_detour = relative_detour

    return max_detour


def demonstrate_fig6(net: Network):
    """
    Reproduces the detouredness calculation from Section 4.2 (Fig. 6).
    Route 1: A->B->C->D->E   expected φ = 0.5
    Route 2: A->B->D->E      expected φ = 0.0
    Route 3: A->D->E         expected φ = 0.25
    """
    routes = {
        "Route 1 (A->B->C->D->E)": [0, 1, 2, 3, 4],
        "Route 2 (A->B->D->E)":    [0, 1, 3, 4],
        "Route 3 (A->D->E)":       [0, 3, 4],
    }
    expected = {"Route 1 (A->B->C->D->E)": 0.5,
                "Route 2 (A->B->D->E)": 0.0,
                "Route 3 (A->D->E)": 0.25}

    print("=" * 55)
    print("Fig. 6 Local Detouredness Verification (Section 4.2)")
    print("=" * 55)
    net.compute_all_pairs_shortest_paths()
    for name, route in routes.items():
        phi = local_detouredness(route, net)
        exp = expected[name]
        status = "✓" if abs(phi - exp) < 1e-9 else "✗"
        cost = net.route_cost(route)
        print(f"  {status} {name}")
        print(f"      Total cost = {cost:.1f}  |  φ = {phi:.4f}  (expected {exp})")
    print()


def show_segment_breakdown(route: List[int], route_name: str, net: Network):
    """
    Verbose breakdown of local detouredness for every segment of a route.
    Useful for debugging and understanding the measure.
    """
    print(f"  Segment breakdown for {route_name}:")
    print(f"  {'Segment':12s}  {'Used cost':10s}  {'Min cost':10s}  {'Detour':8s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}")
    max_d = 0.0
    for i in range(len(route)):
        for j in range(i + 1, len(route)):
            a, b = route[i], route[j]
            used = sum(net.get_link_cost(route[k], route[k+1])
                       for k in range(i, j))
            mn = net.shortest_path_cost(a, b)
            d = max(0.0, (used - mn) / mn) if mn > 0 else 0.0
            max_d = max(max_d, d)
            marker = " <-- MAX" if abs(d - max_d) < 1e-9 and d > 0 else ""
            node_names = "ABCDE"
            seg = f"({node_names[a]},{node_names[b]})"
            print(f"  {seg:12s}  {used:10.2f}  {mn:10.2f}  {d:8.4f}{marker}")
    print(f"  => φ = {max_d:.4f}\n")

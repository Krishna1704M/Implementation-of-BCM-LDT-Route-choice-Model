"""
Part 4: Branch-and-Bound Route Generation (Section 5.2)
=========================================================
Generates ALL simple routes from origin to destination satisfying:
  1. Global cost bound:       route cost  <=  τ * c_min
  2. Local detour threshold:  φ(route)    <   γ

Uses Depth-First Search with branch pruning:
  - Prune branch if current cost + SP-to-dest > τ * c_min_OD
  - Prune branch if local detouredness of partial route > γ

Memoises all-pairs shortest paths for efficiency (as per Section 5.2).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from typing import List, Tuple, Dict, Optional
from network import Network
from detouredness import local_detouredness_incremental


def generate_routes(
    net: Network,
    origin: int,
    destination: int,
    tau: float,
    gamma: float,
    verbose: bool = False
) -> List[Tuple[List[int], float, float]]:
    """
    Branch-and-bound DFS to find all routes from origin to destination
    satisfying the global cost bound τ and local detour threshold γ.

    Parameters
    ----------
    net         : Network (must have all-pairs SP pre-computed)
    origin      : origin node ID
    destination : destination node ID
    tau         : relative global cost bound (>1)
    gamma       : local detour threshold (>0)
    verbose     : print debug info

    Returns
    -------
    List of (route_nodes, route_cost, local_detouredness)
    """
    # Ensure shortest paths are pre-computed
    if not net._sp_cache:
        net.compute_all_pairs_shortest_paths()

    # Minimum-cost route (reference for global bound)
    c_min = net.shortest_path_cost(origin, destination)
    if c_min == float('inf'):
        print(f"  No path from {origin} to {destination}")
        return []

    cost_bound = tau * c_min   # absolute cost budget

    results = []
    n_pruned_cost   = [0]
    n_pruned_detour = [0]

    def dfs(path: List[int], current_cost: float, current_detour: float):
        """
        Recursive DFS with branch-and-bound.
        path           : nodes visited so far (starts with [origin])
        current_cost   : cumulative cost of path so far
        current_detour : max local detouredness of path so far
        """
        node = path[-1]

        if node == destination:
            # Valid complete route found
            results.append((list(path), current_cost, current_detour))
            return

        # Try extending to each unvisited neighbor
        for next_node in net.graph.successors(node):
            if next_node in path:
                continue  # avoid cycles (simple routes only)

            # Cost of moving to next_node
            edge_cost = net.get_link_cost(node, next_node)
            new_cost  = current_cost + edge_cost

            # ── Bounding check 1: global cost ──────────────────────────
            # current cost + best possible remaining cost > bound?
            sp_to_dest = net.shortest_path_cost(next_node, destination)
            if new_cost + sp_to_dest > cost_bound + 1e-9:
                n_pruned_cost[0] += 1
                continue

            # ── Bounding check 2: local detouredness ───────────────────
            # Update detouredness incrementally (only checking new segments
            # ending at next_node — O(|path|) not O(|path|^2))
            new_detour = local_detouredness_incremental(
                path, next_node, current_detour, net)

            if new_detour >= gamma:
                n_pruned_detour[0] += 1
                continue

            # Both checks passed: extend the branch
            path.append(next_node)
            dfs(path, new_cost, new_detour)
            path.pop()

    # Start DFS from origin
    dfs([origin], 0.0, 0.0)

    if verbose:
        print(f"  Routes found:         {len(results)}")
        print(f"  Pruned (cost):        {n_pruned_cost[0]}")
        print(f"  Pruned (detour):      {n_pruned_detour[0]}")
        print(f"  c_min = {c_min:.3f},  cost bound = {cost_bound:.3f}")

    return results


def print_routes_table(routes, c_min: float, tau: float, gamma: float,
                       node_labels: Optional[Dict] = None):
    """Pretty-print a table of generated routes."""
    def node_name(n):
        if node_labels:
            return node_labels.get(n, str(n))
        return str(n)

    print(f"  {'#':<4} {'Route':<30} {'Cost':>8} {'Rel.Cost':>9} {'φ':>8} {'Active?':>8}")
    print(f"  {'-'*4} {'-'*30} {'-'*8} {'-'*9} {'-'*8} {'-'*8}")
    for idx, (route, cost, phi) in enumerate(routes):
        route_str = " -> ".join(node_name(n) for n in route)
        rel_cost  = cost / c_min if c_min > 0 else 1.0
        active    = "yes" if (cost <= tau * c_min + 1e-9 and phi < gamma) else "no"
        print(f"  {idx+1:<4} {route_str:<30} {cost:8.2f} {rel_cost:9.3f} "
              f"{phi:8.4f} {active:>8}")


def demonstrate_fig7_routes(x: float = 10.0,
                             tau: float = 2.0, gamma: float = 1.0):
    """
    Generate routes for the Fig. 7 small network and show how γ/τ
    control which routes are active.
    """
    import sys
    sys.path.insert(0, '/home/claude/bcm_ldt')
    from network import build_fig7_network

    net = build_fig7_network(x)
    net.compute_all_pairs_shortest_paths()

    print(f"\nFig. 7 Network (x={x}): Route generation with τ={tau}, γ={gamma}")
    print("-" * 60)
    routes = generate_routes(net, origin=0, destination=3,
                             tau=tau, gamma=gamma, verbose=True)
    node_labels = {0: "O", 1: "hub1", 2: "hub2", 3: "D"}
    c_min = net.shortest_path_cost(0, 3)
    print_routes_table(routes, c_min, tau, gamma, node_labels)
    return routes

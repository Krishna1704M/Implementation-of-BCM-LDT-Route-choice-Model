"""
Part 5: Stochastic User Equilibrium Solver with BCM-LDT (Algorithm 1)
=======================================================================
Implements the full iterative SUE algorithm from Section 5.2 of the paper.

Four steps per iteration:
  1. Flow allocation   — compute auxiliary flows using BCM-LDT probabilities
  2. Violation removal — redistribute flow from routes violating bounds
  3. Network loading   — update link costs from current flows (BPR/quadratic)
  4. Column generation — branch-and-bound to find any new feasible routes

Convergence: RMSE between consecutive route flow vectors < 10^{-zeta}

MSWA step size (Liu et al. 2009):
    lambda_n = n^z / sum_{k=1}^{n} k^z     (z=0 gives standard MSA)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sys
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

sys.path.insert(0, '/home/claude/bcm_ldt')
from network import Network
from detouredness import local_detouredness
from models import bcm_ldt_probabilities
from route_generation import generate_routes


# ───────────────────────────────────────────────────────────────────
# Data structures
# ───────────────────────────────────────────────────────────────────

@dataclass
class RouteData:
    """All information about a single route for one OD pair."""
    nodes:   List[int]          # node sequence
    cost:    float = 0.0        # current generalised cost
    phi:     float = 0.0        # local detouredness
    flow:    float = 0.0        # current route flow x_mi
    aux_flow: float = 0.0       # auxiliary flow x̄_mi


@dataclass
class ODState:
    """State for one OD movement during SUE iteration."""
    origin:      int
    destination: int
    demand:      float
    routes:      List[RouteData] = field(default_factory=list)

    def route_costs(self):
        return [r.cost for r in self.routes]

    def route_phis(self):
        return [r.phi for r in self.routes]

    def route_flows(self):
        return np.array([r.flow for r in self.routes])

    def total_flow(self):
        return sum(r.flow for r in self.routes)


# ───────────────────────────────────────────────────────────────────
# MSWA step size
# ───────────────────────────────────────────────────────────────────

def mswa_step(n: int, z: float, cache: Dict) -> float:
    """
    Method of Successive Weighted Averages step size (Liu et al. 2009).
    lambda_n = n^z / sum_{k=1}^n k^z
    Cache accumulates the denominator incrementally.
    """
    nz = n ** z
    cache['denom'] = cache.get('denom', 0.0) + nz
    return nz / cache['denom']


# ───────────────────────────────────────────────────────────────────
# Core SUE solver
# ───────────────────────────────────────────────────────────────────

class BCMLDTSolver:
    """
    Iterative SUE solver for the BCM-LDT model.

    Parameters
    ----------
    net      : Network object with link costs and OD demands
    theta1   : cost scaling parameter
    theta2   : detouredness scaling parameter
    tau      : relative global cost bound
    gamma    : local detour threshold
    zeta     : convergence: RMSE < 10^{-zeta}
    max_iter : maximum iterations
    z        : MSWA parameter (0 = standard MSA)
    cost_fn  : 'quadratic' | 'bpr' | 'fixed'
    init_ς   : extra margin added to tau/gamma in Step 0 for initial route generation
    verbose  : print iteration log
    """

    def __init__(self, net: Network,
                 theta1: float = 0.2,
                 theta2: float = 0.2,
                 tau:    float = 1.6,
                 gamma:  float = 0.8,
                 zeta:   float = 3.0,
                 max_iter: int = 200,
                 z:      float = 2.0,
                 cost_fn: str = 'quadratic',
                 init_varsigma: float = 0.3,
                 verbose: bool = True):

        self.net   = net
        self.theta1 = theta1
        self.theta2 = theta2
        self.tau    = tau
        self.gamma  = gamma
        self.zeta   = zeta
        self.max_iter = max_iter
        self.z      = z
        self.cost_fn = cost_fn
        self.init_varsigma = init_varsigma
        self.verbose = verbose

        # Per-OD state
        self.od_states: Dict[Tuple, ODState] = {}

        # Convergence history
        self.rmse_history: List[float] = []
        self.iter_times:   List[float] = []

    # ── Step 0: Initialisation ─────────────────────────────────────

    def initialise(self):
        """
        Step 0: Generate initial choice sets using free-flow costs
        with slightly relaxed bounds (tau + ς, gamma + ς).
        Assign initial flows proportional to BCM-LDT probabilities.
        """
        # Reset to free-flow costs
        self.net.update_link_costs('fixed')
        self.net.compute_all_pairs_shortest_paths()

        tau_init   = self.tau   + self.init_varsigma
        gamma_init = self.gamma + self.init_varsigma

        self.od_states.clear()

        for od in self.net.od_pairs:
            key = (od.origin, od.destination)
            state = ODState(od.origin, od.destination, od.demand)

            routes = generate_routes(
                self.net, od.origin, od.destination,
                tau=tau_init, gamma=gamma_init, verbose=False)

            if not routes:
                # Fallback: at least the shortest path
                sp = self.net.shortest_path(od.origin, od.destination)
                if sp:
                    phi = local_detouredness(sp, self.net)
                    cost = self.net.route_cost(sp)
                    routes = [(sp, cost, phi)]

            for (nodes, cost, phi) in routes:
                state.routes.append(RouteData(nodes=nodes, cost=cost, phi=phi))

            # Initial flows from BCM-LDT probabilities at free-flow
            self._update_route_costs(state)
            P, _, _ = bcm_ldt_probabilities(
                state.route_costs(), state.route_phis(),
                self.theta1, self.theta2, self.tau, self.gamma)
            for i, r in enumerate(state.routes):
                r.flow = od.demand * P[i]

            self.od_states[key] = state

        # Initial network loading
        self._load_network()
        if self.verbose:
            total_routes = sum(len(s.routes) for s in self.od_states.values())
            print(f"  Initialised: {len(self.od_states)} OD pairs, "
                  f"{total_routes} routes total")

    # ── Step 1: Flow allocation ────────────────────────────────────

    def _flow_allocation(self, lam: float):
        """
        Compute auxiliary flows from BCM-LDT probabilities,
        then perform MSWA weighted average with current flows.
        """
        for state in self.od_states.values():
            self._update_route_costs(state)

            P, _, _ = bcm_ldt_probabilities(
                state.route_costs(), state.route_phis(),
                self.theta1, self.theta2, self.tau, self.gamma)

            for i, r in enumerate(state.routes):
                r.aux_flow = state.demand * P[i]
                # MSWA averaging
                r.flow = (1 - lam) * r.flow + lam * r.aux_flow

    # ── Step 2: Violation removal ──────────────────────────────────

    def _remove_violations(self):
        """
        For each OD: routes with aux_flow=0 but flow>0 violate
        the bounds. Redistribute their flow to non-violating routes
        proportional to BCM-LDT probabilities.
        """
        for state in self.od_states.values():
            violating_flow = 0.0
            non_violating  = []

            for r in state.routes:
                if r.aux_flow <= 0 and r.flow > 0:
                    violating_flow += r.flow
                    r.flow = 0.0
                else:
                    non_violating.append(r)

            if violating_flow > 1e-10 and non_violating:
                # Redistribute proportional to aux_flow
                total_aux = sum(r.aux_flow for r in non_violating)
                if total_aux > 1e-10:
                    for r in non_violating:
                        r.flow += violating_flow * (r.aux_flow / total_aux)
                else:
                    share = violating_flow / len(non_violating)
                    for r in non_violating:
                        r.flow += share

    # ── Step 3: Network loading ────────────────────────────────────

    def _load_network(self):
        """
        Aggregate route flows to link flows, update link costs.
        """
        # Reset all link flows to zero
        for link in self.net.links.values():
            link.flow = 0.0

        # Assign route flows to links
        for state in self.od_states.values():
            for r in state.routes:
                if r.flow <= 0:
                    continue
                for i in range(len(r.nodes) - 1):
                    u, v = r.nodes[i], r.nodes[i + 1]
                    lid = self.net.link_lookup.get((u, v))
                    if lid is not None:
                        self.net.links[lid].flow += r.flow

        # Recompute link costs
        self.net.update_link_costs(self.cost_fn)
        # Recompute shortest paths with updated costs (for detouredness)
        self.net.compute_all_pairs_shortest_paths()

    # ── Step 4: Column generation ──────────────────────────────────

    def _column_generation(self) -> int:
        """
        For each OD, run branch-and-bound with current costs.
        Add any new routes not already in the choice set.
        Returns total number of new routes added.
        """
        new_routes_added = 0

        for key, state in self.od_states.items():
            existing = set(
                tuple(r.nodes) for r in state.routes)

            new = generate_routes(
                self.net, state.origin, state.destination,
                tau=self.tau, gamma=self.gamma, verbose=False)

            for (nodes, cost, phi) in new:
                t = tuple(nodes)
                if t not in existing:
                    state.routes.append(
                        RouteData(nodes=nodes, cost=cost, phi=phi, flow=0.0))
                    existing.add(t)
                    new_routes_added += 1

        return new_routes_added

    # ── Update route costs from current link costs ─────────────────

    def _update_route_costs(self, state: ODState):
        """Recompute cost and phi for every route in this OD state."""
        for r in state.routes:
            r.cost = self.net.route_cost(r.nodes)
            r.phi  = local_detouredness(r.nodes, self.net)

    # ── RMSE convergence criterion ────────────────────────────────

    def _compute_rmse(self) -> float:
        """RMSE between current flow and auxiliary flow vectors."""
        sq = 0.0
        n  = 0
        for state in self.od_states.values():
            for r in state.routes:
                sq += (r.flow - r.aux_flow) ** 2
                n  += 1
        return np.sqrt(sq / n) if n > 0 else 0.0

    # ── Main solve loop ────────────────────────────────────────────

    def solve(self) -> Dict:
        """
        Run the BCM-LDT SUE algorithm (Algorithm 1).
        Returns a results dictionary with convergence info and flows.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  BCM-LDT SUE Solver")
            print(f"  θ₁={self.theta1}, θ₂={self.theta2}, τ={self.tau}, γ={self.gamma}")
            print(f"  Convergence: RMSE < 10^{{-{self.zeta}}}")
            print(f"{'='*60}")

        t_start = time.time()
        self.initialise()

        mswa_cache = {}
        converged  = False

        for n in range(1, self.max_iter + 1):
            t_iter = time.time()

            lam = mswa_step(n, self.z, mswa_cache)

            # Step 1: flow allocation
            self._flow_allocation(lam)

            # Step 2: violation removal
            self._remove_violations()

            # Step 3: network loading
            self._load_network()

            # Step 4: column generation
            new_cols = self._column_generation()

            # Convergence check
            rmse = self._compute_rmse()
            self.rmse_history.append(rmse)
            self.iter_times.append(time.time() - t_iter)

            if self.verbose and (n <= 10 or n % 10 == 0):
                total_routes = sum(len(s.routes)
                                   for s in self.od_states.values())
                print(f"  Iter {n:4d} | RMSE={rmse:.6f} | "
                      f"λ={lam:.4f} | routes={total_routes} | "
                      f"+cols={new_cols}")

            if rmse < 10 ** (-self.zeta) and new_cols == 0:
                converged = True
                if self.verbose:
                    print(f"\n  ✓ Converged at iteration {n}  "
                          f"(RMSE={rmse:.2e})")
                break

        elapsed = time.time() - t_start
        if self.verbose:
            if not converged:
                print(f"\n  ✗ Did not converge in {self.max_iter} iterations")
            print(f"  Total time: {elapsed:.2f}s")

        return self._build_results(converged, elapsed)

    # ── Results packaging ─────────────────────────────────────────

    def _build_results(self, converged: bool, elapsed: float) -> Dict:
        results = {
            'converged':    converged,
            'elapsed':      elapsed,
            'rmse_history': self.rmse_history,
            'iter_times':   self.iter_times,
            'od_states':    self.od_states,
            'link_flows':   {lid: lnk.flow
                             for lid, lnk in self.net.links.items()},
            'link_costs':   {lid: lnk.cost
                             for lid, lnk in self.net.links.items()},
        }
        return results

    # ── Summary printing ─────────────────────────────────────────

    def print_summary(self, results: Dict):
        print(f"\n{'─'*60}")
        print("  Flow Summary per OD pair")
        print(f"{'─'*60}")
        for key, state in results['od_states'].items():
            c_min = min(r.cost for r in state.routes)
            print(f"\n  OD ({state.origin}→{state.destination})  "
                  f"demand={state.demand:.1f}")
            print(f"  {'Route':<25} {'Flow':>8} {'Cost':>8} "
                  f"{'Rel.Cost':>9} {'φ':>8}")
            print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*9} {'-'*8}")
            for r in sorted(state.routes, key=lambda x: -x.flow):
                if r.flow < 0.01:
                    continue
                rel = r.cost / c_min if c_min > 0 else 1.0
                nodes_str = "→".join(str(n) for n in r.nodes)
                print(f"  {nodes_str:<25} {r.flow:8.1f} {r.cost:8.2f} "
                      f"{rel:9.3f} {r.phi:8.4f}")
        print()

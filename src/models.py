"""
Part 3: BCM-LDT Choice Probability Model (Equation 17)
========================================================
Implements the Bounded Choice Model with Local Detour Threshold.

The probability of route i for OD m is:

    P_mi = Q1_mi * Q2_mi  /  sum_j (Q1_mj * Q2_mj)

where:
    Q1_mi = (exp(-θ1 * (c_mi - τ * c_min)) - 1)+    [cost-BCM, Eq. 14]
    Q2_mi = (exp(-θ2 * (φ_mi - γ))         - 1)+    [detour-BCM, Eq. 16]
    (·)+ = max(0, ·)

Parameters
----------
θ1  : travel cost scaling (sensitivity to cost relative to bound)
θ2  : detouredness scaling (sensitivity to local detour relative to threshold)
τ   : relative global cost bound  (route excluded if c_mi >= τ * c_min)
γ   : local detour threshold       (route excluded if φ_mi >= γ)
"""

import numpy as np
from typing import List, Tuple, Dict
import sys, os

# Adds the Project BTM_demo_Results folder to Python's search path
# so imports like "from network import Network" work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def relu(x: float) -> float:
    """ReLU: max(0, x)"""
    return max(0.0, x)


# ---------------------------------------------------------------------------
# Individual BCM components
# ---------------------------------------------------------------------------

def cost_bcm_score(c_mi: float, c_min: float,
                   theta1: float, tau: float) -> float:
    """
    Q1_mi — cost-BCM component (Eq. 14).

    Returns (exp(-θ1*(c_mi - τ*c_min)) - 1)+
    Route gets zero if c_mi >= τ * c_min.
    """
    return relu(np.exp(-theta1 * (c_mi - tau * c_min)) - 1.0)


def detour_bcm_score(phi_mi: float, theta2: float, gamma: float) -> float:
    """
    Q2_mi — detour-BCM component (Eq. 16).

    Returns (exp(-θ2*(φ_mi - γ)) - 1)+
    Route gets zero if φ_mi >= γ.
    """
    return relu(np.exp(-theta2 * (phi_mi - gamma)) - 1.0)


# ---------------------------------------------------------------------------
# Main BCM-LDT probability function
# ---------------------------------------------------------------------------

def bcm_ldt_probabilities(
    costs: List[float],
    detours: List[float],
    theta1: float,
    theta2: float,
    tau: float,
    gamma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute BCM-LDT choice probabilities for a set of routes (Eq. 17).

    Parameters
    ----------
    costs   : list of route costs c_mi
    detours : list of local detouredness values φ_mi
    theta1  : cost scaling parameter
    theta2  : detouredness scaling parameter
    tau     : relative global cost bound (>1)
    gamma   : local detour threshold (>0)

    Returns
    -------
    P   : BCM-LDT probabilities (shape: n_routes)
    Q1  : cost-BCM scores (unnormalised)
    Q2  : detour-BCM scores (unnormalised)
    """
    costs   = np.array(costs,   dtype=float)
    detours = np.array(detours, dtype=float)
    n = len(costs)

    c_min = np.min(costs)   # reference: cheapest route

    Q1 = np.array([cost_bcm_score(costs[i], c_min, theta1, tau)
                   for i in range(n)])
    Q2 = np.array([detour_bcm_score(detours[i], theta2, gamma)
                   for i in range(n)])

    # Joint score: product of both BCM components (Eq. 17 numerator)
    joint = Q1 * Q2

    denom = joint.sum()
    if denom <= 0:
        # All routes excluded — assign to cheapest (fallback)
        P = np.zeros(n)
        P[np.argmin(costs)] = 1.0
    else:
        P = joint / denom

    return P, Q1, Q2


# ---------------------------------------------------------------------------
# Comparison models: MNL, BCM (cost only), PSL
# ---------------------------------------------------------------------------

def mnl_probabilities(costs: List[float], theta: float) -> np.ndarray:
    """
    Standard Multinomial Logit probabilities.
    P_i = exp(-θ * c_i) / Σ exp(-θ * c_j)
    """
    costs = np.array(costs, dtype=float)
    log_num = -theta * costs
    log_num -= log_num.max()   # numerical stability
    exp_num = np.exp(log_num)
    return exp_num / exp_num.sum()


def bcm_cost_only(costs: List[float], theta: float,
                  tau: float) -> np.ndarray:
    """
    BCM with only global cost bound (Eq. 14, no detour component).
    """
    costs = np.array(costs, dtype=float)
    c_min = costs.min()
    Q = np.array([relu(np.exp(-theta * (c - tau * c_min)) - 1.0)
                  for c in costs])
    denom = Q.sum()
    if denom <= 0:
        P = np.zeros(len(costs))
        P[np.argmin(costs)] = 1.0
        return P
    return Q / denom


def psl_probabilities(costs: List[float],
                      paths: List[List],
                      theta: float,
                      beta: float) -> np.ndarray:
    """
    Path Size Logit probabilities (Ben-Akiva & Ramming 1998).
    PS_i = Σ_{a in i} (L_a / L_i) * (1 / Σ_{j: a in j} 1)
    P_i = exp(θ*(V_i + β*ln(PS_i))) / Σ_j exp(θ*(V_j + β*ln(PS_j)))

    Simple version: compute path size by link overlap.
    paths: list of route node-sequences
    """
    costs = np.array(costs, dtype=float)
    n = len(paths)

    # Build link sets for each route
    def get_links(path):
        return [(path[k], path[k+1]) for k in range(len(path)-1)]

    route_links = [set(get_links(p)) for p in paths]
    route_costs = costs.copy()

    PS = np.zeros(n)
    for i, (links_i, c_i) in enumerate(zip(route_links, route_costs)):
        ps = 0.0
        for link in links_i:
            # Count how many routes share this link
            sharing = sum(1 for links_j in route_links if link in links_j)
            ps += (1.0 / sharing)   # simplified (assumes uniform link costs)
        PS[i] = ps / len(links_i) if links_i else 1.0

    PS = np.clip(PS, 1e-10, None)
    util = -theta * costs + beta * np.log(PS)
    util -= util.max()
    exp_util = np.exp(util)
    return exp_util / exp_util.sum()


# ---------------------------------------------------------------------------
# Reproduce Fig. 4 comparison table
# ---------------------------------------------------------------------------

def reproduce_fig4():
    """
    Reproduce the comparison table from Fig. 4 (Section 1.2).
    3-route toy network, Scenario A (top of Fig. 4):
      Route 1: cost=210  (large local detour via top node)
      Route 2: cost=200  (mid route)
      Route 3: cost=201  (bottom direct route)
    """
    print("=" * 65)
    print("Fig. 4 Choice Probability Comparison (Scenario A: bottom=201)")
    print("=" * 65)

    # Route costs (Scenario A from paper)
    costs_A = [210.0, 200.0, 201.0]

    # Local detouredness:
    # Route 1: takes a 10-cost detour on a segment where direct is 0
    #          actually: top branch adds 10 cost on a sub-segment
    #          φ1 ≈ (210-200)/200 = 0.05 globally, but locally:
    #          segment O->mid vs. O->mid->top->mid is the detour
    #          The paper shows Route 1 has large local detour (100/100=1.0 ratio locally)
    # For simplicity we use φ values matching the paper's description:
    # Route 1 local detour = 10/100 = 0.1 (10 cost detour on a 100-cost segment)
    # Actually from Fig 4 network: top branch is 10 extra vs direct hub->D (100)
    # so φ1 = 10/100 = 0.1. Route 2,3 φ=0.
    detours_A = [0.1, 0.0, 0.0]   # φ for each route

    # Parameters from paper caption
    theta1 = 0.01   # logit cost scaling
    tau    = 1.5    # relative surplus cost bound
    theta2 = 0.2    # local detouredness scaling
    gamma  = 1.0    # local detour threshold (so φ1=0.1 < γ, all routes active)
    beta   = 1.0    # PSL path size scaling

    # Deterministic (only min-cost route used)
    det_A = np.array([0.0, 0.5, 0.5])   # tie between routes 2&3

    # MNL
    mnl_A = mnl_probabilities(costs_A, theta=theta1)

    # BCM (cost only)
    bcm_A = bcm_cost_only(costs_A, theta=theta1, tau=tau)

    # BCM-LDT
    P_A, Q1_A, Q2_A = bcm_ldt_probabilities(
        costs_A, detours_A, theta1, theta2, tau, gamma)

    # PSL (routes as node sequences — simplified)
    paths_A = [
        [0, 1, 2, 3],   # Route 1: O->hub->top->D
        [0, 1, 3],       # Route 2: O->hub->D
        [0, 3],          # Route 3: O->D direct
    ]
    psl_A = psl_probabilities(costs_A, paths_A, theta=theta1, beta=beta)

    print(f"\n{'Route':<8} {'Det':>8} {'MNL':>8} {'PSL':>8} {'BCM':>8} {'BCM-LDT':>10}")
    print("-" * 55)
    for r in range(3):
        print(f"  {r+1:<6} {det_A[r]:8.2f} {mnl_A[r]:8.2f} "
              f"{psl_A[r]:8.2f} {bcm_A[r]:8.2f} {P_A[r]:10.2f}")

    # Scenario B: bottom route cost = 200
    print(f"\n{'='*65}")
    print("Fig. 4 Scenario B (bottom route cost = 200)")
    print("=" * 65)
    costs_B = [210.0, 200.0, 200.0]
    det_B  = np.array([0.0, 0.0, 1.0])
    mnl_B  = mnl_probabilities(costs_B, theta=theta1)
    bcm_B  = bcm_cost_only(costs_B, theta=theta1, tau=tau)
    P_B, _, _ = bcm_ldt_probabilities(
        costs_B, detours_A, theta1, theta2, tau, gamma)
    psl_B = psl_probabilities(costs_B, paths_A, theta=theta1, beta=beta)

    print(f"\n{'Route':<8} {'Det':>8} {'MNL':>8} {'PSL':>8} {'BCM':>8} {'BCM-LDT':>10}")
    print("-" * 55)
    for r in range(3):
        print(f"  {r+1:<6} {det_B[r]:8.2f} {mnl_B[r]:8.2f} "
              f"{psl_B[r]:8.2f} {bcm_B[r]:8.2f} {P_B[r]:10.2f}")
    print()
    print("Expected from paper: BCM-LDT assigns 0 to Route 1 (large local detour)")
    print("when γ is set tight enough. Above uses γ=1.0 (loose) to show all routes.")
    print("Set γ=0.05 to see Route 1 excluded:\n")

    # Tight threshold to exclude Route 1
    gamma_tight = 0.05
    P_tight, _, _ = bcm_ldt_probabilities(
        costs_A, detours_A, theta1, theta2, tau, gamma_tight)
    print(f"  γ={gamma_tight}: P = {np.round(P_tight, 3)}")
    print()

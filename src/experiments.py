"""
Main Experiments: Parts 1–4
============================
Reproduces key results from the paper on toy networks:
  - Fig. 6: Local detouredness verification
  - Fig. 4: Model comparison table (Det / MNL / PSL / BCM / BCM-LDT)
  - Fig. 8: BCM-LDT probabilities vs γ (small network, x=10)
  - Fig. 9: BCM-LDT probabilities vs γ AND τ
  - Fig. 10: BCM-LDT probabilities vs x (varying overlap)
  - Fig. 12: Q1, Q2, P components vs x
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#import sys
#sys.path.insert(0, '/home/claude/bcm_ldt')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap

from network import (build_fig6_network, build_fig7_network,
                     build_fig16_network, Network)
from detouredness import (local_detouredness, demonstrate_fig6,
                          show_segment_breakdown)
from models import (bcm_ldt_probabilities, mnl_probabilities,
                    bcm_cost_only, psl_probabilities, reproduce_fig4)
from route_generation import generate_routes, print_routes_table


# ── Matplotlib style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 120,
    'lines.linewidth': 2.2,
})
COLORS = ['#c0392b', '#e67e22', '#27ae60']   # Route 1, 2, 3


# ===========================================================================
# Helper: compute detouredness for Fig. 7 routes as x varies
# ===========================================================================

def fig7_route_detours(x: float):
    """
    For the Fig. 7 network with parameter x, return:
      costs    = [c1, c2, c3]
      detours  = [φ1, φ2, φ3]
    
    Route 1: O->hub1->D via link2, cost = x + (25-x) = 25
    Route 2: O->hub2->D via link3, cost = x + (20-x) = 20  (cheapest)
    Route 3: O->D direct,          cost = 35
    
    φ calculations:
      Route 1: worst segment is hub1->D (cost 25-x) vs D direct from hub (20-x)
               detour = (25-x - (20-x)) / (20-x) = 5/(20-x)
      Route 2: no detour, φ2 = 0
      Route 3: worst segment is O->D (35) vs O->D shortest (20)
               detour = (35-20)/20 = 0.75  [constant regardless of x]
    """
    c1, c2, c3 = 25.0, 20.0, 35.0
    # φ1 = (25-x - (20-x)) / (20-x) but routes share link 1 (cost x),
    # so the local detour on the SHARED segment O->hub is:
    # Route1 sub-route O->hub1->D costs 25, but O->hub2->D costs 20
    # => φ1 = (25 - 20) / 20 = 0.25? No – routes have different hubs.
    # Paper says detouredness of Route 1 = (25-x-(20-x))/(20-x) = 5/(20-x)
    phi1 = 5.0 / (20.0 - x) if (20.0 - x) > 0 else float('inf')
    phi2 = 0.0
    phi3 = 0.75   # (35-20)/20, constant
    return [c1, c2, c3], [phi1, phi2, phi3]


# ===========================================================================
# Fig. 8: BCM-LDT probabilities vs γ  (x=10, τ=2)
# ===========================================================================

def plot_fig8(save=True):
    x       = 10.0
    tau     = 2.0
    theta1  = theta2 = 0.1
    gammas  = np.linspace(0.01, 2.05, 300)

    costs, _ = fig7_route_detours(x)
    phi1 = 5.0 / (20.0 - x)   # = 0.5 at x=10
    phi3 = 0.75
    detours = [phi1, 0.0, phi3]

    probs = {0: [], 1: [], 2: []}
    for g in gammas:
        P, _, _ = bcm_ldt_probabilities(costs, detours, theta1, theta2, tau, g)
        for r in range(3):
            probs[r].append(P[r])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r, (color, label) in enumerate(zip(COLORS, ['Route 1', 'Route 2', 'Route 3'])):
        ax.plot(gammas, probs[r], color=color, label=label)

    ax.axvline(phi1, ls=':', color='gray', lw=1.2, alpha=0.7, label=f'φ₁={phi1:.2f}')
    ax.axvline(phi3, ls='--', color='gray', lw=1.2, alpha=0.7, label=f'φ₃={phi3:.2f}')
    ax.set_xlabel('γ  (local detour threshold)')
    ax.set_ylabel('Probability')
    ax.set_title(f'Fig. 8 — BCM-LDT Probabilities vs γ\n(τ={tau}, θ₁=θ₂={theta1}, x={x})')
    ax.legend(loc='center right')
    ax.set_xlim(0, 2.05)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        path = 'results/fig8_gamma_sweep.png'
        plt.savefig(path, dpi=130, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig


# ===========================================================================
# Fig. 10: BCM-LDT probabilities vs x  (γ=1, τ=2)
# ===========================================================================

def plot_fig10(save=True):
    gamma   = 1.0
    tau     = 2.0
    theta1  = theta2 = 0.1
    xs      = np.linspace(0.01, 19.99, 300)

    probs = {0: [], 1: [], 2: []}
    for x in xs:
        costs, detours = fig7_route_detours(x)
        P, _, _ = bcm_ldt_probabilities(costs, detours, theta1, theta2, tau, gamma)
        for r in range(3):
            probs[r].append(P[r])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r, (color, label) in enumerate(zip(COLORS, ['Route 1', 'Route 2', 'Route 3'])):
        ax.plot(xs, probs[r], color=color, label=label)

    # Mark x=15: where φ1 = 5/(20-x) = 1 = γ
    x_cross = 15.0
    ax.axvline(x_cross, ls=':', color='#7f8c8d', lw=1.3,
               label=f'x={x_cross} (φ₁ hits γ=1)')
    ax.set_xlabel('x')
    ax.set_ylabel('Probability')
    ax.set_title(f'Fig. 10 — BCM-LDT Probabilities vs x\n(γ={gamma}, τ={tau}, θ₁=θ₂={theta1})')
    ax.legend()
    ax.set_xlim(0, 20)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        path = 'results/fig10_x_sweep.png'
        plt.savefig(path, dpi=130, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig


# ===========================================================================
# Fig. 12: Q1, Q2, P components vs x
# ===========================================================================

def plot_fig12(save=True):
    gamma   = 1.0
    tau     = 2.0
    theta1  = theta2 = 0.1
    xs      = np.linspace(0.01, 19.99, 300)

    route_labels = ['Route 1', 'Route 2', 'Route 3']
    styles = ['-', '--', ':']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    titles = ['Q¹ (cost-BCM)', 'Q² (detour-BCM)', 'P (BCM-LDT)']

    all_Q1, all_Q2, all_P = [], [], []
    for x in xs:
        costs, detours = fig7_route_detours(x)
        P, Q1, Q2 = bcm_ldt_probabilities(costs, detours, theta1, theta2, tau, gamma)
        all_Q1.append(Q1)
        all_Q2.append(Q2)
        all_P.append(P)

    all_Q1 = np.array(all_Q1)
    all_Q2 = np.array(all_Q2)
    all_P  = np.array(all_P)

    for ax, data, title in zip(axes, [all_Q1, all_Q2, all_P], titles):
        for r, (color, label, ls) in enumerate(zip(COLORS, route_labels, styles)):
            ax.plot(xs, data[:, r], color=color, label=label, ls=ls)
        ax.axvline(15.0, ls=':', color='gray', lw=1.0, alpha=0.6)
        ax.set_xlabel('x')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, 20)

    axes[0].set_ylabel('Score / Probability')
    fig.suptitle('Fig. 12 — Q¹, Q², P Components vs x\n'
                 f'(γ={gamma}, τ={tau}, θ₁=θ₂={theta1})', y=1.02)
    plt.tight_layout()
    if save:
        path = 'results/fig12_components.png'
        plt.savefig(path, dpi=130, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig


# ===========================================================================
# Fig. 9 surface: probabilities vs γ AND τ
# ===========================================================================

def plot_fig9(save=True):
    x      = 10.0
    theta1 = theta2 = 0.1
    gammas = np.linspace(0.01, 3.0, 60)
    taus   = np.linspace(1.01, 3.0, 60)

    costs, _ = fig7_route_detours(x)
    phi1 = 5.0 / (20.0 - x)
    phi3 = 0.75
    detours = [phi1, 0.0, phi3]

    P_grids = [np.zeros((len(taus), len(gammas))) for _ in range(3)]

    for i, tau in enumerate(taus):
        for j, g in enumerate(gammas):
            P, _, _ = bcm_ldt_probabilities(costs, detours, theta1, theta2, tau, g)
            for r in range(3):
                P_grids[r][i, j] = P[r]

    T, G = np.meshgrid(taus, gammas, indexing='ij')

    fig = plt.figure(figsize=(14, 4.5))
    route_labels = ['Route 1', 'Route 2', 'Route 3']
    for r, (label, cmap_name) in enumerate(
            zip(route_labels, ['Reds', 'Blues', 'Greens'])):
        ax = fig.add_subplot(1, 3, r + 1, projection='3d')
        surf = ax.plot_surface(T, G, P_grids[r],
                               cmap=cmap_name, alpha=0.85, linewidth=0)
        ax.set_xlabel('τ', labelpad=6)
        ax.set_ylabel('γ', labelpad=6)
        ax.set_zlabel('P', labelpad=6)
        ax.set_title(f'{label}', fontsize=10)
        ax.set_zlim(0, 1)
        ax.tick_params(labelsize=7)

    fig.suptitle('Fig. 9 — BCM-LDT Probability Surfaces vs γ and τ\n'
                 f'(x={x}, θ₁=θ₂={theta1})', y=1.01)
    plt.tight_layout()
    if save:
        path = 'results/fig9_surfaces.png'
        plt.savefig(path, dpi=110, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig


# ===========================================================================
# Run everything
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "█" * 65)
    print("  BCM-LDT Implementation — Parts 1–4")
    print("  Local detouredness + BCM-LDT probabilities on toy networks")
    print("█" * 65 + "\n")

    # ── Part 2: verify detouredness on Fig. 6 ─────────────────────────────
    net6 = build_fig6_network()
    net6.compute_all_pairs_shortest_paths()
    demonstrate_fig6(net6)

    # Verbose breakdown of Route 1
    show_segment_breakdown([0, 1, 2, 3, 4], "Route 1 (A->B->C->D->E)", net6)

    # ── Part 3: Fig. 4 comparison table ───────────────────────────────────
    reproduce_fig4()

    # ── Part 4: Route generation on Fig. 7 ────────────────────────────────
    print("=" * 65)
    print("Part 4: Branch-and-Bound Route Generation on Fig. 7 Network")
    print("=" * 65)
    net7 = build_fig7_network(x=10.0)
    net7.compute_all_pairs_shortest_paths()
    c_min7 = net7.shortest_path_cost(0, 3)

    for g in [0.3, 0.6, 1.0, 2.0]:
        print(f"\n  τ=2.0, γ={g}:")
        routes = generate_routes(net7, origin=0, destination=3,
                                 tau=2.0, gamma=g, verbose=False)
        labels = {0: "O", 1: "hub1", 2: "hub2", 3: "D"}
        print_routes_table(routes, c_min7, tau=2.0, gamma=g, node_labels=labels)

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Generating Figures...")
    print("=" * 65)
    plot_fig8()
    plot_fig10()
    plot_fig12()
    plot_fig9()

    print("\n✓  All Parts 1–4 complete.")
    print("   Output figures saved to results/")

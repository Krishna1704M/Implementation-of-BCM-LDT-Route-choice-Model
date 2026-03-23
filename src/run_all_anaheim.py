"""
Part 6: Complete Numerical Experiments — Real Anaheim Network
==============================================================
Reproduces all paper experiments using the actual Anaheim TNTP data.

Experiments:
  [1] Table 2  — 3-route congested network comparison
  [2] Fig 14   — Choice set size vs gamma (OD 4->7)
  [3] Fig 15   — Link usage for different gamma values
  [4] Fig 17   — SUE convergence (subset of ODs)
  [5] Fig 18   — Choice set distribution
  [6] Fig 19   — Average choice set size vs gamma
  [7] Fig 21   — Equilibrated link flows
  [8] Fig 22   — Flow difference BCM-LDT vs DUE
  [9] Fig 23   — Route probability scatter
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from collections import defaultdict

from network import build_fig16_network
from detouredness import local_detouredness
from models import bcm_ldt_probabilities, bcm_cost_only, mnl_probabilities
from route_generation import generate_routes
from sue_solver import BCMLDTSolver
from anaheim_loader import load_anaheim

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 120, 'lines.linewidth': 2.0,
})
COLORS = ['#c0392b','#e67e22','#27ae60','#2980b9','#8e44ad']
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# [1] TABLE 2  — 3-route congested small network
# ═══════════════════════════════════════════════════════════════════

def run_model1(net, tau=1.3, gamma_cutoff=0.5, theta1=0.01):
    net.update_link_costs('fixed'); net.compute_all_pairs_shortest_paths()
    all_routes = generate_routes(net,0,3,tau=2.0,gamma=2.0)
    allowed = [(n,c,p) for n,c,p in all_routes if p < gamma_cutoff] or all_routes[:1]
    demand = net.od_pairs[0].demand
    flows  = np.array([demand/len(allowed)]*len(allowed))
    for it in range(500):
        for lnk in net.links.values(): lnk.flow=0.0
        for i,(nodes,_,_) in enumerate(allowed):
            for k in range(len(nodes)-1):
                lid=net.link_lookup.get((nodes[k],nodes[k+1]))
                if lid: net.links[lid].flow+=flows[i]
        net.update_link_costs('quadratic')
        costs=[net.route_cost(n) for n,_,_ in allowed]
        probs=bcm_cost_only(costs,theta=theta1,tau=tau)
        aux=demand*probs; lam=1/(it+1)
        nf=(1-lam)*flows+lam*aux
        if np.sqrt(np.mean((nf-flows)**2))<1e-4: flows=nf; break
        flows=nf
    costs=[net.route_cost(n) for n,_,_ in allowed]
    phis=[local_detouredness(n,net) for n,_,_ in allowed]
    cm=min(costs)
    return [(allowed[i][0],flows[i],costs[i],costs[i]/cm,phis[i]) for i in range(len(allowed))]

def run_model2(net, tau=1.3, gamma_cutoff=0.5, theta1=0.01):
    net.update_link_costs('fixed'); net.compute_all_pairs_shortest_paths()
    all_routes=generate_routes(net,0,3,tau=2.0,gamma=2.0)
    demand=net.od_pairs[0].demand; flows=np.array([demand/len(all_routes)]*len(all_routes))
    for it in range(500):
        for lnk in net.links.values(): lnk.flow=0.0
        for i,(nodes,_,_) in enumerate(all_routes):
            for k in range(len(nodes)-1):
                lid=net.link_lookup.get((nodes[k],nodes[k+1]))
                if lid: net.links[lid].flow+=flows[i]
        net.update_link_costs('quadratic'); net.compute_all_pairs_shortest_paths()
        costs=[net.route_cost(n) for n,_,_ in all_routes]
        phis=[local_detouredness(n,net) for n,_,_ in all_routes]
        probs=bcm_cost_only(costs,theta=theta1,tau=tau); aux=demand*probs
        for i,phi in enumerate(phis):
            if phi>=gamma_cutoff: aux[i]=0.0
        tot=aux.sum()
        if tot>0: aux=aux/tot*demand
        lam=1/(it+1); nf=(1-lam)*flows+lam*aux
        if np.sqrt(np.mean((nf-flows)**2))<1e-4: flows=nf; break
        flows=nf
    costs=[net.route_cost(n) for n,_,_ in all_routes]
    phis=[local_detouredness(n,net) for n,_,_ in all_routes]
    cm=min(costs)
    return [(all_routes[i][0],flows[i],costs[i],costs[i]/cm,phis[i]) for i in range(len(all_routes))]

def run_model3(net, tau=1.3, gamma=0.5, theta1=0.01, theta2=1.0):
    solver=BCMLDTSolver(net,theta1=theta1,theta2=theta2,tau=tau,gamma=gamma,
                        zeta=3,max_iter=300,z=0,cost_fn='quadratic',
                        init_varsigma=0.4,verbose=False)
    results=solver.solve()
    state=list(results['od_states'].values())[0]
    costs=[r.cost for r in state.routes]; cm=min(costs)
    return [(r.nodes,r.flow,r.cost,r.cost/cm,r.phi) for r in state.routes]

def run_mnl(net, theta=0.01):
    net.update_link_costs('fixed'); net.compute_all_pairs_shortest_paths()
    all_routes=generate_routes(net,0,3,tau=5.0,gamma=10.0)
    demand=net.od_pairs[0].demand; flows=np.array([demand/len(all_routes)]*len(all_routes))
    for it in range(500):
        for lnk in net.links.values(): lnk.flow=0.0
        for i,(nodes,_,_) in enumerate(all_routes):
            for k in range(len(nodes)-1):
                lid=net.link_lookup.get((nodes[k],nodes[k+1]))
                if lid: net.links[lid].flow+=flows[i]
        net.update_link_costs('quadratic')
        costs=[net.route_cost(n) for n,_,_ in all_routes]
        probs=mnl_probabilities(costs,theta=theta)
        aux=demand*probs; lam=1/(it+1)
        nf=(1-lam)*flows+lam*aux
        if np.sqrt(np.mean((nf-flows)**2))<1e-4: flows=nf; break
        flows=nf
    costs=[net.route_cost(n) for n,_,_ in all_routes]
    phis=[local_detouredness(n,net) for n,_,_ in all_routes]
    cm=min(costs)
    return [(all_routes[i][0],flows[i],costs[i],costs[i]/cm,phis[i]) for i in range(len(all_routes))]

def table2():
    print("\n"+"="*70)
    print("Table 2 — 3-Route Congested Network  (τ=1.3, γ=0.5, demand=5000)")
    print("="*70)
    def fresh():
        n=build_fig16_network(); n.compute_all_pairs_shortest_paths(); return n
    rmap={(0,1,3):'R1: O→hub→D(detour)',(0,1,2,3):'R2: O→hub→D(upper)',(0,3):'R3: O→D(direct)'}
    models={'Model 1 (pre-filter)':run_model1(fresh()),
            'Model 2 (hard cutoff)':run_model2(fresh()),
            'Model 3 (BCM-LDT)':run_model3(fresh()),
            'MNL SUE':run_mnl(fresh())}
    for name,res in models.items():
        print(f"\n  {name}:")
        print(f"  {'Route':<22}{'Flow':>8}{'Cost':>9}{'RelCost':>9}{'φ':>8}")
        print(f"  {'-'*22}{'-'*8}{'-'*9}{'-'*9}{'-'*8}")
        for nodes,flow,cost,rel,phi in res:
            rname=rmap.get(tuple(nodes),str(nodes))
            print(f"  {rname:<22}{flow:8.1f}{cost:9.2f}{rel:9.3f}{phi:8.4f}")

def plot_table2_bar():
    def fresh():
        n=build_fig16_network(); n.compute_all_pairs_shortest_paths(); return n
    rmap={(0,1,3):0,(0,1,2,3):1,(0,3):2}
    def gf(res):
        m=[0,0,0]
        for nodes,flow,*_ in res:
            if tuple(nodes) in rmap: m[rmap[tuple(nodes)]]=flow
        return m
    data={'Model 1\n(pre-filter)':gf(run_model1(fresh())),
          'Model 2\n(hard cutoff)':gf(run_model2(fresh())),
          'Model 3\n(BCM-LDT)':gf(run_model3(fresh())),
          'MNL SUE':gf(run_mnl(fresh()))}
    xlbls=['R1: detour\nO→hub→D','R2: upper\nO→hub→D','R3: direct\nO→D']
    x=np.arange(3); w=0.18
    fig,ax=plt.subplots(figsize=(10,5.5))
    for i,(model,flows) in enumerate(data.items()):
        ax.bar(x+(i-1.5)*w,flows,w,label=model,
               color=COLORS[i],alpha=0.85,edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(xlbls,fontsize=10)
    ax.set_ylabel('Route Flow (vehicles)')
    ax.set_title('Table 2 — Flow Comparison Across Models\n(τ=1.3, γ=0.5, demand=5000)')
    ax.legend(loc='upper right',fontsize=9); ax.grid(True,alpha=0.3,axis='y')
    ax.set_ylim(0,5800)
    m3=data['Model 3\n(BCM-LDT)']
    ax.annotate('BCM-LDT: detour\nroute near zero',
                xy=(x[0]+0.5*w,m3[0]+30),xytext=(0.5,2800),
                arrowprops=dict(arrowstyle='->',color='#2ecc71',lw=1.5),
                fontsize=8,color='#27ae60')
    plt.tight_layout()
    plt.savefig(f'{OUT}/table2_bar.png',dpi=130,bbox_inches='tight')
    print(f"  Saved: table2_bar.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# [2] FIG 14  — Choice set size vs gamma, Anaheim OD 4->7
# ═══════════════════════════════════════════════════════════════════

def fig14_choiceset(net, tau=1.6):
    gammas=[0.1,0.2,0.3,0.4,0.5,0.6]
    sizes=[]
    print(f"\n  Fig 14: choice set sizes for OD 4->7 (τ={tau})")
    for g in gammas:
        t0=time.time()
        routes=generate_routes(net,4,7,tau=tau,gamma=g,verbose=False)
        sizes.append(len(routes))
        print(f"    γ={g:.1f}: {len(routes):6d} routes  ({time.time()-t0:.1f}s)")

    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.semilogy(gammas,sizes,'o-',color='#2980b9',lw=2.2,ms=7)
    # Add tau reference lines
    ax.axvline(tau-1,ls='--',color='gray',lw=1.2,alpha=0.7,label=f'τ-1={tau-1}')
    ax.set_xlabel('γ  (local detour threshold)')
    ax.set_ylabel('Choice set size  (log scale)')
    ax.set_title(f'Fig. 14 — Choice Set Size vs γ\n(τ={tau}, Anaheim OD: 4→7)')
    ax.legend(); ax.grid(True,alpha=0.3,which='both')
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig14_choiceset.png',dpi=130,bbox_inches='tight')
    print(f"  Saved: fig14_choiceset.png"); plt.close()
    return dict(zip(gammas,sizes))


# ═══════════════════════════════════════════════════════════════════
# [3] FIG 15  — Link usage for different gamma (OD 4->7)
# ═══════════════════════════════════════════════════════════════════

def fig15_link_usage(net, tau=1.6, gammas_plot=[0.1,0.4,0.8,2.2]):
    import networkx as nx
    # Get node positions from graph (just use indices as proxy)
    nodes_list = sorted(net.graph.nodes())
    pos = nx.spring_layout(net.graph, seed=42, k=0.3)  # layout for visualization

    fig,axes=plt.subplots(2,2,figsize=(14,10))
    axes=axes.flatten()

    for ax_idx,(g,ax) in enumerate(zip(gammas_plot,axes)):
        routes=generate_routes(net,4,7,tau=tau,gamma=g,verbose=False)
        if not routes:
            ax.set_title(f'γ={g}: no routes'); continue

        # Aggregate link probabilities
        costs=[r[1] for r in routes]
        phis=[r[2] for r in routes]
        P,_,_=bcm_ldt_probabilities(costs,phis,0.2,0.2,tau,g)

        link_prob=defaultdict(float)
        for i,(nodes,c,phi) in enumerate(routes):
            for k in range(len(nodes)-1):
                link_prob[(nodes[k],nodes[k+1])]+=P[i]

        if link_prob:
            max_p=max(link_prob.values())
            for (u,v),prob in link_prob.items():
                if prob<0.001 or u not in pos or v not in pos: continue
                x=[pos[u][0],pos[v][0]]; y=[pos[u][1],pos[v][1]]
                lw=0.5+5.0*prob/max_p
                ax.plot(x,y,color='#c0392b',lw=lw,alpha=0.6+0.4*prob/max_p)

        ax.scatter(*zip(*[pos[n] for n in pos if n in net.graph.nodes()]),
                   s=1,color='#2c3e50',alpha=0.3)
        # Highlight OD
        if 4 in pos: ax.scatter(*pos[4],s=100,color='green',zorder=5,label='O')
        if 7 in pos: ax.scatter(*pos[7],s=100,color='red',zorder=5,label='D')
        ax.set_title(f'γ={g}  |  Choice set: {len(routes)} routes')
        ax.axis('off')
        if ax_idx==0: ax.legend(loc='upper left',fontsize=8)

    fig.suptitle(f'Fig. 15 — BCM-LDT Link Probabilities for Different γ\n'
                 f'(τ={tau}, Anaheim OD 4→7)',y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig15_link_usage.png',dpi=110,bbox_inches='tight')
    print(f"  Saved: fig15_link_usage.png"); plt.close()


# ═══════════════════════════════════════════════════════════════════
# [4] FIG 17  — SUE convergence on OD subset
# ═══════════════════════════════════════════════════════════════════

def fig17_convergence(tau=1.6, theta1=0.2, theta2=0.2,
                       gammas=[0.3,0.5,0.7], max_iter=40):
    # Use top-demand OD pairs for SUE
    net_all=load_anaheim()
    top_ods=sorted(net_all.od_pairs,key=lambda x:-x.demand)[:20]
    od_subset=[(o.origin,o.destination) for o in top_ods]

    rmse_curves={}
    for g in gammas:
        print(f"\n  SUE convergence: γ={g} ...")
        net=load_anaheim(od_subset=od_subset)
        solver=BCMLDTSolver(net,theta1=theta1,theta2=theta2,
                            tau=tau,gamma=g,zeta=2,max_iter=max_iter,
                            z=2,cost_fn='bpr',init_varsigma=0.3,verbose=True)
        solver.solve()
        rmse_curves[g]=solver.rmse_history

    fig,ax=plt.subplots(figsize=(8,4.5))
    cmap=plt.cm.plasma
    for i,g in enumerate(sorted(rmse_curves)):
        rmse=rmse_curves[g]
        color=cmap(i/max(len(rmse_curves)-1,1))
        ax.semilogy(range(1,len(rmse)+1),rmse,color=color,label=f'γ={g}',lw=2.2)
        ax.scatter(len(rmse),rmse[-1],color=color,s=60,zorder=5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Route Flow RMSE (log scale)')
    ax.set_title('Fig. 17 — BCM-LDT SUE Convergence\n'
                 f'(τ={tau}, top-20 OD pairs, Anaheim network)')
    ax.legend(loc='upper right'); ax.grid(True,alpha=0.3,which='both')
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig17_convergence.png',dpi=130,bbox_inches='tight')
    print(f"  Saved: fig17_convergence.png"); plt.close()


# ═══════════════════════════════════════════════════════════════════
# [5+6] FIG 18 & 19  — Choice set distribution and avg size
# ═══════════════════════════════════════════════════════════════════

def fig18_19_distributions(net, tau=1.6, sample_ods=None,
                             gammas=[0.2,0.4,0.6,0.8,1.0]):
    if sample_ods is None:
        sample_ods=net.od_pairs[:25]

    results={}
    print(f"\n  Figs 18&19: choice set distribution across {len(sample_ods)} ODs")
    for g in gammas:
        all_routes=[]; sizes=[]
        for od in sample_ods:
            r=generate_routes(net,od.origin,od.destination,
                              tau=tau,gamma=g,verbose=False)
            sizes.append(len(r)); all_routes.extend(r)
        results[g]={'sizes':sizes,'routes':all_routes,
                    'avg':np.mean(sizes),'median':np.median(sizes)}
        print(f"    γ={g:.1f}: avg={np.mean(sizes):.1f}  "
              f"median={np.median(sizes):.0f}  total={len(all_routes)}")

    # ── Fig 18: CDF ──────────────────────────────────────────────
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    cmap=plt.cm.viridis
    for i,g in enumerate(gammas):
        routes=results[g]['routes']
        if not routes: continue
        color=cmap(i/max(len(gammas)-1,1))
        costs=sorted(r[1] for r in routes)
        cm=costs[0] if costs else 1.0
        rel=[c/cm for c in costs]
        phis=sorted(r[2] for r in routes)
        cdf=np.arange(1,len(rel)+1)/len(rel)
        axes[0].plot(rel,cdf,color=color,label=f'γ={g}',lw=1.8)
        cdf2=np.arange(1,len(phis)+1)/len(phis)
        axes[1].plot(phis,cdf2,color=color,label=f'γ={g}',lw=1.8)

    axes[0].set_xlabel('Relative route cost (c/c_min)')
    axes[0].set_ylabel('CDF'); axes[0].set_title('Route Cost Distribution')
    axes[0].legend(fontsize=8); axes[0].grid(True,alpha=0.3)
    axes[1].set_xlabel('Local detouredness φ')
    axes[1].set_ylabel('CDF'); axes[1].set_title('Detouredness Distribution')
    axes[1].legend(fontsize=8); axes[1].grid(True,alpha=0.3)
    fig.suptitle('Fig. 18 — Choice Set Distributions  (Anaheim network)',y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig18_distribution.png',dpi=130,bbox_inches='tight')
    print(f"  Saved: fig18_distribution.png"); plt.close()

    # ── Fig 19: avg size ─────────────────────────────────────────
    avgs=[results[g]['avg'] for g in gammas]
    tots=[len(results[g]['routes']) for g in gammas]
    fig,ax1=plt.subplots(figsize=(8,4.5))
    ax2=ax1.twinx()
    l1,=ax1.semilogy(gammas,avgs,'o-',color='#2980b9',lw=2.2,ms=7,
                     label='Avg choice set size')
    l2,=ax2.semilogy(gammas,tots,'s--',color='#c0392b',lw=1.8,ms=6,
                     label='Total routes (all sampled ODs)')
    ax1.set_xlabel('γ  (local detour threshold)')
    ax1.set_ylabel('Avg choice set size (log)',color='#2980b9')
    ax2.set_ylabel('Total routes (log)',color='#c0392b')
    ax1.tick_params(axis='y',labelcolor='#2980b9')
    ax2.tick_params(axis='y',labelcolor='#c0392b')
    ax1.set_title('Fig. 19 — Choice Set Size vs γ  (Anaheim network)')
    ax1.legend([l1,l2],[l.get_label() for l in [l1,l2]],loc='upper left')
    ax1.grid(True,alpha=0.3,which='both')
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig19_avg_choiceset.png',dpi=130,bbox_inches='tight')
    print(f"  Saved: fig19_avg_choiceset.png"); plt.close()


# ═══════════════════════════════════════════════════════════════════
# [7] FIG 21  — Equilibrated link flows
# ═══════════════════════════════════════════════════════════════════

def fig21_flows(sue_results, net, gamma, tau):
    import networkx as nx
    pos=nx.spring_layout(net.graph,seed=42,k=0.5)
    link_flows=sue_results['link_flows']
    max_flow=max(link_flows.values()) if link_flows else 1.0

    fig,ax=plt.subplots(figsize=(10,8))
    for lid,lnk in net.links.items():
        flow=link_flows.get(lid,0.0)
        if flow<1.0 or lnk.u not in pos or lnk.v not in pos: continue
        x=[pos[lnk.u][0],pos[lnk.v][0]]
        y=[pos[lnk.u][1],pos[lnk.v][1]]
        lw=0.3+4.5*(flow/max_flow)
        alpha=0.3+0.7*(flow/max_flow)
        color=plt.cm.YlOrRd(flow/max_flow)
        ax.plot(x,y,color=color,lw=lw,alpha=alpha,solid_capstyle='round')

    sm=plt.cm.ScalarMappable(cmap='YlOrRd',
                              norm=mcolors.Normalize(0,max_flow))
    sm.set_array([]); plt.colorbar(sm,ax=ax,label='Link flow (veh)',shrink=0.7)
    ax.set_title(f'Fig. 21 — BCM-LDT Equilibrated Link Flows\n'
                 f'(γ={gamma}, τ={tau}, Anaheim network)')
    ax.axis('off'); plt.tight_layout()
    plt.savefig(f'{OUT}/fig21_flows.png',dpi=130,bbox_inches='tight')
    print(f"  Saved: fig21_flows.png"); plt.close()


# ═══════════════════════════════════════════════════════════════════
# [8] FIG 22  — Flow difference BCM-LDT vs DUE
# ═══════════════════════════════════════════════════════════════════

def fig22_diff(flows_bcmldt, flows_due, net, gamma):
    import networkx as nx
    pos=nx.spring_layout(net.graph,seed=42,k=0.5)

    diffs={lid: flows_bcmldt.get(lid,0)-flows_due.get(lid,0)
           for lid in net.links}
    max_d=max(abs(d) for d in diffs.values()) or 1.0

    fig,ax=plt.subplots(figsize=(10,8))
    for lid,lnk in net.links.items():
        d=diffs.get(lid,0.0)
        if abs(d)<0.5 or lnk.u not in pos or lnk.v not in pos: continue
        x=[pos[lnk.u][0],pos[lnk.v][0]]
        y=[pos[lnk.u][1],pos[lnk.v][1]]
        lw=0.5+4.0*abs(d)/max_d
        color='#c0392b' if d>0 else '#2980b9'
        ax.plot(x,y,color=color,lw=lw,alpha=0.75,solid_capstyle='round')

    legend_elems=[Patch(facecolor='#c0392b',label='BCM-LDT > DUE  (more flow)'),
                  Patch(facecolor='#2980b9',label='BCM-LDT < DUE  (less flow)')]
    ax.legend(handles=legend_elems,loc='upper right',fontsize=9)
    ax.set_title(f'Fig. 22 — Flow Difference: BCM-LDT vs DUE\n(γ={gamma}, τ=1.6)')
    ax.axis('off'); plt.tight_layout()
    plt.savefig(f'{OUT}/fig22_flow_diff.png',dpi=130,bbox_inches='tight')
    print(f"  Saved: fig22_flow_diff.png"); plt.close()


# ═══════════════════════════════════════════════════════════════════
# [9] FIG 23  — Route probability scatter
# ═══════════════════════════════════════════════════════════════════

def fig23_scatter(net, od_origin=4, od_dest=7,
                   gamma=0.8, tau=1.6, theta1=0.2, theta2=0.2):
    net.update_link_costs('fixed'); net.compute_all_pairs_shortest_paths()
    routes=generate_routes(net,od_origin,od_dest,tau=tau,gamma=gamma,verbose=False)
    if not routes: print("  No routes for Fig 23"); return

    costs=np.array([r[1] for r in routes])
    phis=np.array([r[2] for r in routes])
    cm=costs.min(); rel=costs/cm
    P,_,_=bcm_ldt_probabilities(list(costs),list(phis),theta1,theta2,tau,gamma)

    fig,ax=plt.subplots(figsize=(7,5.5))
    sc=ax.scatter(rel,phis,c=P,cmap='RdYlGn',s=25,alpha=0.8,
                  norm=mcolors.Normalize(0,P.max()))
    plt.colorbar(sc,ax=ax,label='BCM-LDT Probability')
    ax.axvline(tau,ls='--',color='gray',lw=1.3,alpha=0.8,label=f'τ={tau}')
    ax.axhline(gamma,ls=':',color='gray',lw=1.3,alpha=0.8,label=f'γ={gamma}')
    ax.set_xlabel('Relative Global Detour  (c / c_min)')
    ax.set_ylabel('Local Detouredness  φ')
    ax.set_title(f'Fig. 23 — Route Probabilities at BCM-LDT\n'
                 f'Anaheim OD {od_origin}→{od_dest}  |  {len(routes)} routes  '
                 f'(γ={gamma}, τ={tau})')
    ax.legend(fontsize=8); ax.grid(True,alpha=0.25)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig23_scatter.png',dpi=130,bbox_inches='tight')
    print(f"  Saved: fig23_scatter.png"); plt.close()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__=='__main__':
    print("\n"+"█"*65)
    print("  BCM-LDT — Parts 5 & 6: Complete Experiments")
    print("  Using real Anaheim network (416 nodes, 914 links)")
    print("█"*65)

    # ── [1] Table 2 ───────────────────────────────────────────────
    print("\n[1/9] Table 2: 3-route congested network")
    table2()
    plot_table2_bar()

    # ── Load Anaheim with congested costs ─────────────────────────
    print("\n  Loading Anaheim (congested costs)...")
    net = load_anaheim(use_congested_costs=True)
    net.update_link_costs('fixed')
    net.compute_all_pairs_shortest_paths()
    print("  All-pairs SP computed.")

    # ── [2] Fig 14 ────────────────────────────────────────────────
    print("\n[2/9] Fig 14: Choice set size vs gamma")
    fig14_choiceset(net, tau=1.6)

    # ── [3] Fig 15 ────────────────────────────────────────────────
    print("\n[3/9] Fig 15: Link usage visualization")
    fig15_link_usage(net, tau=1.6, gammas_plot=[0.1, 0.3, 0.5, 0.6])

    # ── [5+6] Figs 18 & 19 ───────────────────────────────────────
    print("\n[4/9] Figs 18 & 19: Choice set distributions")
    sample = net.od_pairs[:30]
    fig18_19_distributions(net, tau=1.6, sample_ods=sample,
                            gammas=[0.2, 0.4, 0.6, 0.8, 1.0])

    # ── [9] Fig 23 ────────────────────────────────────────────────
    print("\n[5/9] Fig 23: Route probability scatter (OD 4->7)")
    fig23_scatter(net, od_origin=4, od_dest=7,
                  gamma=0.8, tau=1.6, theta1=0.2, theta2=0.2)

    # ── [4] Fig 17: SUE convergence ───────────────────────────────
    print("\n[6/9] Fig 17: SUE convergence")
    fig17_convergence(tau=1.6, theta1=0.2, theta2=0.2,
                      gammas=[0.4, 0.6, 0.8], max_iter=60)

    # ── [7+8] Figs 21+22: SUE flow distribution ──────────────────
    print("\n[7/9] Figs 21 & 22: Equilibrated flows + DUE comparison")
    top20=[(o.origin,o.destination) for o in
           sorted(net.od_pairs,key=lambda x:-x.demand)[:20]]

    print("  Running BCM-LDT SUE (γ=0.8)...")
    net08=load_anaheim(od_subset=top20)
    solver08=BCMLDTSolver(net08,theta1=0.2,theta2=0.2,tau=1.6,gamma=0.5,
                          zeta=2,max_iter=60,z=2,cost_fn='bpr',
                          init_varsigma=0.3,verbose=True)
    res08=solver08.solve()
    fig21_flows(res08,net08,gamma=0.5,tau=1.6)

    print("  Running DUE approx (γ≈0, τ≈1)...")
    net_due=load_anaheim(od_subset=top20)
    solver_due=BCMLDTSolver(net_due,theta1=0.2,theta2=0.2,tau=1.02,gamma=0.05,
                            zeta=2,max_iter=60,z=2,cost_fn='bpr',
                            init_varsigma=0.1,verbose=False)
    res_due=solver_due.solve()
    fig22_diff(res08['link_flows'],res_due['link_flows'],net08,gamma=0.8)

    print("\n"+"✓"*5+"  ALL PARTS 1–6 COMPLETE  "+"✓"*5)
    print(f"  All figures saved to {OUT}/")

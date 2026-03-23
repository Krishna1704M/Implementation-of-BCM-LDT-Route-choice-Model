"""
Anaheim Network Loader (with congested cost support)
======================================================
"""
import sys, re, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)

from network import Network

DEFAULT_NET   = os.path.join(BASE_DIR, 'data', 'Anaheim_net.tntp')
DEFAULT_TRIPS = os.path.join(BASE_DIR, 'data', 'Anaheim_trips.tntp')
DEFAULT_FLOW  = os.path.join(BASE_DIR, 'data', 'Anaheim_flow.tntp')


#**DELETE these lines** (all comments and the explanation block):

# This finds the folder where anaheim_loader.py is sitting
# On your Mac this will automatically become:
# /Users/krishnashivajiraonagamwad/Desktop/class slides/
# Behavioural Travel Modelling/Project_BTM/Project BTM_demo_Results/

# This adds that folder to Python's search path
# so it can find network.py, detouredness.py etc.

# This builds the path to your data folder automatically
# On your Mac it becomes:
# /Users/krishnashivajiraonagamwad/Desktop/.../Project BTM_demo_Results/data/Anaheim_net.tntp

#What Python actually sees when it runs this on your Mac:

#BASE_DIR      →  /Users/...
#DEFAULT_NET   →  /Users/...
#DEFAULT_TRIPS →  /Users/...
#DEFAULT_FLOW  →  /Users/...

def _parse_flow_file(flow_path):
    flow_data = {}
    with open(flow_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    u,v = int(parts[0]),int(parts[1])
                    flow_data[(u,v)] = float(parts[3])
                except: pass
    return flow_data

def load_anaheim(net_path=None, trips_path=None, flow_path=None,
                 od_subset=None, use_congested_costs=True,
                 travel_time_weight=1.0, distance_weight=0.5):
    net_path   = net_path   or DEFAULT_NET
    trips_path = trips_path or DEFAULT_TRIPS
    flow_path  = flow_path  or DEFAULT_FLOW

    flow_data = {}
    if use_congested_costs and os.path.exists(flow_path):
        flow_data = _parse_flow_file(flow_path)

    net = Network("Anaheim")
    link_id = 1

    with open(net_path) as f:
        in_data = False
        for line in f:
            line = line.strip()
            if '~' in line and 'init_node' in line.lower():
                in_data = True; continue
            if not in_data or not line or line.startswith('<'): continue
            parts = line.rstrip(';').split()
            if len(parts) < 5: continue
            try:
                u=int(parts[0]); v=int(parts[1])
                cap=float(parts[2]); length_ft=float(parts[3]); fft=float(parts[4])
            except: continue
            dist_km  = length_ft / 3280.84
            cong_time = flow_data.get((u,v), fft)
            gen_cost  = max(travel_time_weight*cong_time + distance_weight*dist_km, 0.001)
            net.add_node(u); net.add_node(v)
            net.add_link(link_id, u, v, free_flow=round(gen_cost,6), capacity=max(cap,1.0))
            link_id += 1

    od_set = set(od_subset) if od_subset else None
    od_count = 0
    with open(trips_path) as f:
        content = f.read()
    for block in re.split(r'Origin\s+', content, flags=re.IGNORECASE)[1:]:
        lines  = block.strip().split('\n')
        origin = int(lines[0].strip())
        for line in lines[1:]:
            for ds, dm in re.findall(r'(\d+)\s*:\s*([\d.]+)', line):
                dest=int(ds); demand=float(dm)
                if demand<=0: continue
                if od_set and (origin,dest) not in od_set: continue
                net.add_od(origin, dest, demand); od_count+=1

    print(f"  Anaheim: {net.graph.number_of_nodes()} nodes, "
          f"{net.graph.number_of_edges()} links, {od_count} ODs "
          f"({'congested' if flow_data else 'free-flow'})")
    return net

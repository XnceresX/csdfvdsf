import yaml

with open('configs/topology.yaml') as f:
    topo = yaml.safe_load(f)

print('=== EXIT ZONES ANALYSIS ===')
for cam_id, cfg in topo['cameras'].items():
    print(f'{cam_id} ({cfg["name"]}):')
    neighbors = cfg.get('neighbors', [])
    for z in cfg.get('exit_zones', []):
        is_term = z.get('is_terminal', False)
        term_name = z.get('terminal_name')
        handoff = z.get('handoff_cam')
        box = z.get('box')
        print(f'  - Zone: "{z["name"]}" | box={box} | is_terminal={is_term} | term={term_name} | handoff={handoff}')
        if not is_term and not handoff:
            print(f'    ERROR: Non-terminal zone has no handoff_cam!')
        if not is_term and handoff not in neighbors:
            print(f'    WARNING: Handoff cam {handoff} is NOT in neighbors list {neighbors}!')
        if is_term and not term_name:
            print(f'    ERROR: Terminal zone has no terminal_name!')

print('\n=== NEIGHBOR & BOUNDS SYMMETRY CHECK ===')
cameras = topo['cameras']
all_edges = set()
for cam_id, cfg in cameras.items():
    neighbors = set(cfg.get('neighbors', []))
    transit_bounds = cfg.get('transit_bounds_sec', {})
    for n in neighbors:
        all_edges.add((cam_id, n))
        # check reverse neighbor
        rev_cfg = cameras.get(n, {})
        rev_neighbors = set(rev_cfg.get('neighbors', []))
        if cam_id not in rev_neighbors:
            print(f'ASYMMETRY: {cam_id} has neighbor {n}, but {n} does not have neighbor {cam_id}!')
        # check bounds
        b_fwd = transit_bounds.get(n)
        b_rev = rev_cfg.get('transit_bounds_sec', {}).get(cam_id)
        if b_fwd is None:
            print(f'MISSING BOUND: {cam_id} -> {n} has no transit_bounds_sec!')
        if b_rev is None:
            print(f'MISSING BOUND: {n} -> {cam_id} has no transit_bounds_sec!')
        elif b_fwd != b_rev:
            print(f'BOUND MISMATCH: {cam_id} -> {n} is {b_fwd}, but {n} -> {cam_id} is {b_rev}!')

print('\n=== EXIT ZONE HANDOFF COVERAGE ===')
for cam_id, cfg in cameras.items():
    neighbors = set(cfg.get('neighbors', []))
    handoff_targets = {z.get('handoff_cam') for z in cfg.get('exit_zones', []) if not z.get('is_terminal')}
    missing = neighbors - handoff_targets
    if missing:
        print(f'{cam_id} has neighbors {missing} with NO exit zone handoff configured!')

print('\n=== TERMINAL DESTINATIONS LIST ===')
terminals = set()
for cam_id, cfg in cameras.items():
    for z in cfg.get('exit_zones', []):
        if z.get('is_terminal'):
            terminals.add((z.get('terminal_name'), cam_id))
for t, c in sorted(terminals):
    print(f'Terminal: "{t}" on camera {c}')


import heapq
from collections import deque

def solve_maze_algo(grid, start, end, algo):
    cols = len(grid)
    rows = len(grid[0])
    start = tuple(start)
    end = tuple(end)
    
    # Directions: Up, Down, Left, Right
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    visited_order = []
    path = []
    
    if algo == 'bfs':
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, current_path = queue.popleft()
            visited_order.append(current)
            
            if current == end:
                path = current_path
                break
            
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                
                if 0 <= nx < cols and 0 <= ny < rows and grid[nx][ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), current_path + [(nx, ny)]))

    elif algo == 'dfs':
        stack = [(start, [start])]
        visited = {start}
        
        while stack:
            current, current_path = stack.pop()
            visited_order.append(current)
            
            if current == end:
                path = current_path
                break
            
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                
                if 0 <= nx < cols and 0 <= ny < rows and grid[nx][ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append(((nx, ny), current_path + [(nx, ny)]))

    elif algo == 'astar':
        # Priority Queue stores (f_score, current_node, path)
        # f_score = g_score + h_score
        pq = [(0, start, [start])]
        visited = set()
        g_scores = {start: 0}
        
        while pq:
            _, current, current_path = heapq.heappop(pq)
            
            if current in visited:
                continue
            visited.add(current)
            visited_order.append(current)
            
            if current == end:
                path = current_path
                break
            
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                
                if 0 <= nx < cols and 0 <= ny < rows and grid[nx][ny] == 0:
                    new_g = g_scores[current] + 1
                    if (nx, ny) not in g_scores or new_g < g_scores[(nx, ny)]:
                        g_scores[(nx, ny)] = new_g
                        h = abs(nx - end[0]) + abs(ny - end[1]) # Manhattan distance
                        f = new_g + h
                        heapq.heappush(pq, (f, (nx, ny), current_path + [(nx, ny)]))

    elif algo == 'greedy':
        # Priority Queue stores (h_score, current_node, path)
        pq = [(0, start, [start])]
        visited = set()
        
        while pq:
            _, current, current_path = heapq.heappop(pq)
            
            if current in visited:
                continue
            visited.add(current)
            visited_order.append(current)
            
            if current == end:
                path = current_path
                break
            
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                
                if 0 <= nx < cols and 0 <= ny < rows and grid[nx][ny] == 0 and (nx, ny) not in visited:
                    h = abs(nx - end[0]) + abs(ny - end[1])
                    heapq.heappush(pq, (h, (nx, ny), current_path + [(nx, ny)]))

    return {
        'visited': visited_order,
        'path': path
    }

    return {'path': [], 'error': 'No path found'}

import random

def generate_city_map(rows, cols):
    grid = [[0 for _ in range(rows)] for _ in range(cols)]
    # 0: Grass, 1: Road, 2: Building
    
    # Create a grid of roads
    block_size = 4
    for x in range(cols):
        for y in range(rows):
            if x % block_size == 0 or y % block_size == 0:
                grid[x][y] = 1 # Road
            else:
                if random.random() > 0.3:
                    grid[x][y] = 2 # Building
                else:
                    grid[x][y] = 0 # Grass/Park

    # Generate Bins
    bins = []
    for x in range(cols):
        for y in range(rows):
            if grid[x][y] == 1: # On road
                # Check neighbors for building
                has_building = False
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < cols and 0 <= ny < rows and grid[nx][ny] == 2:
                        has_building = True
                        break
                
                if has_building and random.random() < 0.05:
                    bins.append({
                        'x': x, 
                        'y': y, 
                        'level': random.randint(0, 100) # 0-100% full
                    })

    # One-ways (metadata)
    # Make only ~30% of roads one-way to improve connectivity
    one_ways = {}
    for x in range(cols):
        if x % block_size == 0:
            # Vertical road
            if random.random() < 0.3: # 30% chance
                direction = (0, 1) if (x // block_size) % 2 == 0 else (0, -1)
                for y in range(rows):
                    if grid[x][y] == 1:
                        if y % block_size != 0:
                            one_ways[(x,y)] = direction

    for y in range(rows):
        if y % block_size == 0:
            # Horizontal road
            if random.random() < 0.3: # 30% chance
                direction = (1, 0) if (y // block_size) % 2 == 0 else (-1, 0)
                for x in range(cols):
                    if grid[x][y] == 1:
                        if x % block_size != 0:
                            one_ways[(x,y)] = direction

    return {
        'grid': grid,
        'bins': bins,
        'one_ways': [{'x': k[0], 'y': k[1], 'dx': v[0], 'dy': v[1]} for k, v in one_ways.items()]
    }

def generate_parking_lot(rows, cols):
    # 0: Driveway, 1: Wall, 2: Free Spot, 3: Occupied, 4: Reserved
    grid = [[0 for _ in range(rows)] for _ in range(cols)]
    
    # Fill with walls initially? No, fill with driveway (0) and build spots
    # Actually, let's make it an enclosed lot
    for x in range(cols):
        for y in range(rows):
            if x == 0 or x == cols - 1 or y == 0 or y == rows - 1:
                grid[x][y] = 1 # Wall
            else:
                grid[x][y] = 0 # Driveway
                
    # Create lanes and spots
    # Layout: Row of spots, Lane, Row of spots
    # Spot depth = 2 cells? Let's say 1 cell for simplicity on small grid, or 2 for realism.
    # Let's stick to 1 cell = 1 spot unit for now to fit more on screen.
    
    spots = []
    
    # We need lanes. Let's say every 3rd row is a lane.
    # y=1: Spots, y=2: Lane, y=3: Spots, y=4: Spots, y=5: Lane...
    
    for y in range(1, rows - 1):
        # Decide if this row is for spots or lane
        # Pattern: Spot, Lane, Spot, Spot, Lane, Spot...
        # Modulo 3 pattern: 1 (Spot), 2 (Lane), 0 (Spot) -> 1, 2, 0, 1, 2, 0
        
        row_type = y % 3
        
        if row_type != 2: # 2 is Lane (Driveway)
            # This is a spot row
            for x in range(1, cols - 1):
                # Leave some gaps for cross-lanes?
                if x % 8 == 0: 
                    grid[x][y] = 0 # Cross lane
                else:
                    # It's a spot
                    # Randomly assign state
                    r = random.random()
                    if r < 0.4:
                        state = 3 # Occupied
                    elif r < 0.5:
                        state = 4 # Reserved (will refine later)
                    else:
                        state = 2 # Free
                    
                    grid[x][y] = state
                    spots.append({'x': x, 'y': y, 'state': state})
        else:
            # Lane, ensure it's 0
            for x in range(1, cols - 1):
                grid[x][y] = 0

    # Entry point
    start = {'x': 1, 'y': 2} # Start of first lane
    grid[0][2] = 0 # Open entrance
    grid[1][2] = 0
    
    # Refine Reserved Spots
    # Find spots closest to entrance and make them reserved
    spots.sort(key=lambda s: (s['x'] - start['x'])**2 + (s['y'] - start['y'])**2)
    
    # Make top 10% reserved
    num_reserved = max(1, int(len(spots) * 0.1))
    for i in range(len(spots)):
        if i < num_reserved:
            grid[spots[i]['x']][spots[i]['y']] = 4 # Reserved
        elif grid[spots[i]['x']][spots[i]['y']] == 4:
            # If it was randomly reserved but far away, make it free or occupied
            grid[spots[i]['x']][spots[i]['y']] = 2 if random.random() > 0.5 else 3

    return {
        'grid': grid,
        'start': start
    }

def solve_garbage_algo(grid, start, bins, time_of_day, one_ways_list):
    # time_of_day: 'morning', 'afternoon', 'night'
    # Traffic weights
    base_cost = 1
    traffic_cost = 0
    if time_of_day == 'morning': traffic_cost = 5
    elif time_of_day == 'afternoon': traffic_cost = 3
    elif time_of_day == 'night': traffic_cost = 0
    
    # Parse one_ways
    one_ways = {}
    for item in one_ways_list:
        one_ways[(item['x'], item['y'])] = (item['dx'], item['dy'])
        
    # Filter bins and calc total garbage
    active_bins = [b for b in bins if b['level'] > 0]
    total_garbage = sum(b['level'] for b in active_bins)
    
    if not active_bins:
        return {'paths': [], 'stats': {'trucks': 0, 'garbage': 0, 'time': 0}, 'message': 'No trash to collect!'}
        
    # Determine number of trucks (Capacity 100 per truck, but let's say 200 for fewer trucks)
    truck_capacity = 200
    num_trucks = (total_garbage + truck_capacity - 1) // truck_capacity
    num_trucks = max(1, num_trucks)
    
    # Clustering (Simple K-Means-ish or just geometric partitioning)
    # For simplicity and robustness on small grids, let's use K-Means
    # If num_trucks == 1, all bins in one cluster
    
    clusters = [[] for _ in range(num_trucks)]
    if num_trucks == 1:
        clusters[0] = active_bins
    else:
        # Initialize centroids randomly from bins
        centroids = random.sample(active_bins, num_trucks)
        
        # Iterative assignment (just 5 iterations is enough for this scale)
        for _ in range(5):
            clusters = [[] for _ in range(num_trucks)]
            for b in active_bins:
                # Find nearest centroid
                best_dist = float('inf')
                best_c = 0
                for i, c in enumerate(centroids):
                    dist = (b['x'] - c['x'])**2 + (b['y'] - c['y'])**2
                    if dist < best_dist:
                        best_dist = dist
                        best_c = i
                clusters[best_c].append(b)
            
            # Recompute centroids
            new_centroids = []
            for i in range(num_trucks):
                if clusters[i]:
                    avg_x = sum(b['x'] for b in clusters[i]) / len(clusters[i])
                    avg_y = sum(b['y'] for b in clusters[i]) / len(clusters[i])
                    new_centroids.append({'x': avg_x, 'y': avg_y})
                else:
                    # Handle empty cluster (re-init)
                    new_centroids.append(centroids[i]) 
            centroids = new_centroids

    # Solve TSP for each cluster
    all_paths = []
    total_time = 0
    
    cols = len(grid)
    rows = len(grid[0])
    
    def get_path_cost(p1, p2):
        # A*
        start_node = (p1['x'], p1['y'])
        end_node = (p2['x'], p2['y'])
        
        pq = [(0, start_node, [start_node])]
        visited = {start_node: 0}
        
        while pq:
            cost, curr, path = heapq.heappop(pq)
            
            if curr == end_node:
                return cost, path
            
            if cost > visited.get(curr, float('inf')):
                continue
                
            # Get neighbors
            # Relaxed One-Ways: Allow moving against flow but with high cost
            # This ensures connectivity while preferring legal routes
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                
            for dx, dy in neighbors:
                nx, ny = curr[0] + dx, curr[1] + dy
                
                if 0 <= nx < cols and 0 <= ny < rows and grid[nx][ny] == 1: # Must be road
                    move_cost = base_cost + traffic_cost
                    
                    # Check one-way violation
                    if curr in one_ways:
                        allowed_dx, allowed_dy = one_ways[curr]
                        if (dx, dy) != (allowed_dx, allowed_dy):
                            move_cost += 100 # High penalty for going wrong way
                            
                    new_cost = cost + move_cost
                    
                    if new_cost < visited.get((nx, ny), float('inf')):
                        visited[(nx, ny)] = new_cost
                        h = abs(nx - end_node[0]) + abs(ny - end_node[1])
                        heapq.heappush(pq, (new_cost + h, (nx, ny), path + [(nx, ny)]))
                        
        return float('inf'), []

    for cluster_bins in clusters:
        if not cluster_bins: continue
        
        # POIs: Start + Cluster Bins
        # Start is always the depot (start argument)
        pois = [{'x': start[0], 'y': start[1], 'id': -1, 'level': 0}] + [{'x': b['x'], 'y': b['y'], 'id': i, 'level': b['level']} for i, b in enumerate(cluster_bins)]
        n = len(pois)
        
        # Distance Matrix
        dist_matrix = [[float('inf')] * n for _ in range(n)]
        path_matrix = [[[]] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j: 
                    dist_matrix[i][j] = 0
                    continue
                d, p = get_path_cost(pois[i], pois[j])
                dist_matrix[i][j] = d
                path_matrix[i][j] = p
        
        # TSP - Nearest Neighbor
        current_idx = 0
        visited_pois = {0}
        cluster_path = []
        
        # Time Calculation Variables
        current_load = 0
        truck_time = 0
        collection_time_per_bin = 5 # minutes
        
        while len(visited_pois) < n:
            best_dist = float('inf')
            best_next = -1
            
            for j in range(n):
                if j not in visited_pois:
                    if dist_matrix[current_idx][j] < best_dist:
                        best_dist = dist_matrix[current_idx][j]
                        best_next = j
                        
            if best_next == -1:
                break
                
            segment = path_matrix[current_idx][best_next]
            dist = len(segment) - 1 if segment else 0
            
            # Time Calc: Distance * Traffic * Load_Factor
            # Load factor: 1.0 (empty) to 1.2 (full)
            load_factor = 1.0 + (current_load / truck_capacity) * 0.2
            # Traffic factor is implicit in A* cost? No, A* cost was for finding path.
            # Here we want estimated MINUTES.
            # Let's say 1 cell = 100m. Speed = 30km/h / traffic_factor.
            # Simplified: Time = Distance * (1 + Traffic_Level) * Load_Factor
            traffic_level = 0
            if time_of_day == 'morning': traffic_level = 0.5
            elif time_of_day == 'afternoon': traffic_level = 0.3
            
            segment_time = dist * (1 + traffic_level) * load_factor
            truck_time += segment_time
            
            # Collection
            truck_time += collection_time_per_bin
            current_load += pois[best_next]['level']
            
            if cluster_path:
                cluster_path.extend(segment[1:])
            else:
                cluster_path.extend(segment)
                
            visited_pois.add(best_next)
            current_idx = best_next
            
        # Return to depot
        # current_idx is the last bin visited. We need to go back to pois[0] (Depot)
        # We need to calculate path from current_idx to 0
        
        # Check if we are already at depot (unlikely unless no bins)
        if current_idx != 0:
             # We might not have pre-calculated path from current_idx to 0 if we only did Upper Triangle?
             # My dist_matrix loop was `for i in range(n): for j in range(n):` so it's full matrix. Good.
             
             return_dist = dist_matrix[current_idx][0]
             return_path = path_matrix[current_idx][0]
             
             if return_path:
                 if cluster_path:
                     cluster_path.extend(return_path[1:])
                 else:
                     cluster_path.extend(return_path)
                 
                 # Add return time
                 dist = len(return_path) - 1
                 traffic_level = 0
                 if time_of_day == 'morning': traffic_level = 0.5
                 elif time_of_day == 'afternoon': traffic_level = 0.3
                 
                 # Return trip is with empty truck? Or full?
                 # Actually, they collected garbage, so they are FULL.
                 # Load factor is max.
                 load_factor = 1.2 
                 
                 segment_time = dist * (1 + traffic_level) * load_factor
                 truck_time += segment_time
        
        all_paths.append(cluster_path)
        total_time += truck_time

    return {
        'paths': all_paths,
        'stats': {
            'trucks': len(all_paths),
            'garbage': total_garbage,
            'time': int(total_time), # Round to nearest minute
            'algorithm': 'K-Means + TSP + A* (Time-Weighted)'
        }
    }

def solve_parking_algo(grid, start, algo, handicap):
    cols = len(grid)
    rows = len(grid[0])
    start = tuple(start)
    
    # Grid values: 0: driveway, 1: wall, 2: free, 3: occupied, 4: reserved
    # Valid spots: 2 (Free). If handicap, 4 (Reserved) is also valid (and preferred?).
    # Actually, let's say:
    # If handicap: Preferred = 4, then 2.
    # If not handicap: Only 2.
    
    valid_spots = {2}
    if handicap:
        valid_spots.add(4)
        
    # Find all target spots
    targets = []
    for x in range(cols):
        for y in range(rows):
            if grid[x][y] in valid_spots:
                targets.append((x, y))
                
    if not targets:
        return {'path': [], 'error': 'No available parking spots.'}
        
    # BFS/A* to find nearest target
    # Since we want the *closest* spot, BFS is naturally good for this (first target hit is closest).
    # A* with h=distance to nearest target is also good.
    
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    pq = [(0, start, [start])] # cost, current, path
    visited = {start}
    
    while pq:
        cost, current, path = heapq.heappop(pq)
        
        if grid[current[0]][current[1]] in valid_spots:
            # Found a spot!
            # If handicap and we found a regular spot, should we keep looking for a reserved one?
            # For simplicity, let's just find the *nearest* valid spot.
            # If we want to prioritize reserved, we could give them lower "cost" or check them first?
            # But BFS finds nearest by distance.
            # If we want to prioritize reserved spots even if they are further, we need to adjust costs.
            # Let's say: Moving is cost 1. Landing on Reserved is cost 0. Landing on Free is cost 0.
            # If we want reserved to be "better", we can't really do that with simple pathfinding unless we search for ALL spots and compare.
            # Let's just return the nearest valid spot.
            return {'path': path}
            
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            
            if 0 <= nx < cols and 0 <= ny < rows:
                cell = grid[nx][ny]
                # Can traverse 0 (driveway), 2 (free), 4 (reserved).
                # Cannot traverse 1 (wall), 3 (occupied).
                # Wait, can we drive THROUGH a parking spot? Usually yes in a lot.
                # So we can traverse anything except 1 and 3.
                if cell != 1 and cell != 3:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        
                        new_cost = cost + 1
                        h = 0
                        if algo == 'astar':
                            # Distance to nearest target
                            min_dist = min(abs(nx - tx) + abs(ny - ty) for tx, ty in targets)
                            h = min_dist
                            
                        heapq.heappush(pq, (new_cost + h, (nx, ny), path + [(nx, ny)]))
                        
    return {'path': [], 'error': 'No path to a parking spot found.'}

def solve_diagnosis_algo(symptoms, algo, target=None):
    # Knowledge Base
    # Rule format: ([conditions], result)
    rules = [
        (['HighTemp', 'LowPressure'], 'CoolantLeak'),
        (['CoolantLeak'], 'Overheating'),
        (['Overheating'], 'SystemShutdown'),
        (['Vibration', 'Noise'], 'BearingFault'),
        (['BearingFault'], 'MotorFailure'),
        (['MotorFailure'], 'SystemShutdown'),
        (['VoltageSpike'], 'CircuitDamage'),
    ]
    
    if algo == 'forward':
        known_facts = set(symptoms)
        steps = []
        new_fact_added = True
        
        while new_fact_added:
            new_fact_added = False
            for conditions, result in rules:
                if result not in known_facts:
                    if all(c in known_facts for c in conditions):
                        known_facts.add(result)
                        steps.append(f"Rule ({' AND '.join(conditions)} -> {result}) triggered.")
                        new_fact_added = True
                        
        return {
            'inferred': list(known_facts),
            'steps': steps
        }
        
    elif algo == 'backward':
        # Backward Chaining
        # Goal: Verify 'target'
        trace = []
        
        def verify(goal, current_trace):
            if goal in symptoms:
                current_trace.append(f"Fact '{goal}' is observed.")
                return True
            
            # Find rules that infer this goal
            matching_rules = [r for r in rules if r[1] == goal]
            if not matching_rules:
                current_trace.append(f"No rules infer '{goal}' and it is not observed.")
                return False
            
            for conditions, result in matching_rules:
                current_trace.append(f"Checking Rule ({' AND '.join(conditions)} -> {result})")
                all_conditions_met = True
                for c in conditions:
                    if not verify(c, current_trace):
                        all_conditions_met = False
                        break
                
                if all_conditions_met:
                    current_trace.append(f"Rule ({' AND '.join(conditions)} -> {result}) satisfied.")
                    return True
            
            return False

        verified = verify(target, trace)
        return {
            'verified': verified,
            'trace': trace
        }
        
        return {
            'verified': verified,
            'trace': trace
        }
        
    return {'error': 'Invalid algorithm'}

def solve_automation_algo(action_type, sensors, target=None):
    # Sensors: motion, night, temp, user_home
    # Facts generation
    facts = set()
    if sensors.get('motion'): facts.add('MotionDetected')
    if sensors.get('night'): facts.add('NightTime')
    if sensors.get('user_home'): facts.add('UserAtHome')
    if sensors.get('temp', 20) < 18: facts.add('LowTemp')
    
    # Rules: (conditions, result)
    rules = [
        (['MotionDetected', 'NightTime'], 'TurnOn(Lights)'),
        (['LowTemp'], 'TurnOn(Heater)'),
        (['TurnOn(Heater)'], 'IncreaseEnergyUsage'),
        (['IncreaseEnergyUsage'], 'Alert(User)'),
        (['UserAtHome', 'NightTime'], 'Lock(Doors)'),
    ]
    
    if action_type == 'update':
        # Forward Chaining to find all actions
        inferred_actions = []
        known_facts = facts.copy()
        new_fact = True
        
        while new_fact:
            new_fact = False
            for conditions, result in rules:
                if result not in known_facts:
                    if all(c in known_facts for c in conditions):
                        known_facts.add(result)
                        # Only add to "Actions" if it looks like an action (contains '(')
                        if '(' in result:
                            inferred_actions.append(result)
                        new_fact = True
                        
        return {'actions': inferred_actions}
        
    elif action_type == 'explain':
        # Backward Chaining to explain 'target'
        if target not in facts:
            # Check if it can be inferred first
            # (Re-run forward chaining to see if target is true)
            known_facts = facts.copy()
            new_fact = True
            while new_fact:
                new_fact = False
                for conditions, result in rules:
                    if result not in known_facts:
                        if all(c in known_facts for c in conditions):
                            known_facts.add(result)
                            new_fact = True
            
            if target not in known_facts:
                return {'explanation': f"The action '{target}' is NOT currently active, so I cannot explain why it is on."}

        # Now explain
        explanation = []
        
        def explain(goal):
            if goal in facts:
                return f"Sensor '{goal}' is active."
            
            # Find rule that triggered this
            for conditions, result in rules:
                if result == goal:
                    # Check if this rule is actually satisfied (it should be if goal is true)
                    # But we need to find WHICH rule if multiple (here unique)
                    return f"Because {', '.join([explain(c) for c in conditions])}, implying {result}."
            return "Unknown reason."

        # A better explanation builder
        def build_trace(goal):
            if goal in facts:
                return f"observed fact '{goal}'"
            
            for conditions, result in rules:
                if result == goal:
                    cond_explanations = [build_trace(c) for c in conditions]
                    return f"rule ({' AND '.join(conditions)} -> {result}) triggered by {', '.join(cond_explanations)}"
            return "unknown cause"

        return {'explanation': f"{target} is active because {build_trace(target)}."}

    return {'error': 'Invalid action'}

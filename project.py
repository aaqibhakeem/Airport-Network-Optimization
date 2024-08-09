import csv
import heapq
import networkx as nx
from tabulate import tabulate
import plotly.graph_objects as go

def load_airport_coordinates(file_path):
    airports = {}
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            code = row['code']
            name = row['name']
            latitude = float(row['latitude_deg'])
            longitude = float(row['longitude_deg'])
            region = row['region_name']
            municipality = row['municipality']
            
            # Handling duplicate airport codes
            if code in airports:
                suffix = 1
                new_code = f"{code}_{suffix}"
                while new_code in airports:
                    suffix += 1
                    new_code = f"{code}_{suffix}"
                code = new_code

            airports[code] = {
                'name': name,
                'latitude': latitude,
                'longitude': longitude,
                'region': region,
                'municipality': municipality
            }
    return airports

def create_graph_from_airports(airports):
    G = nx.Graph()
    
    # Adding nodes with positions (latitude, longitude)
    for code, data in airports.items():
        G.add_node(code, pos=(data['longitude'], data['latitude']), name=data['name'], region=data['region'], municipality=data['municipality'])
    
    # Adding edges with distances
    airport_codes = list(airports.keys())
    for i in range(len(airport_codes)):
        for j in range(i + 1, len(airport_codes)):
            distance = calculate_custom_distance(G.nodes[airport_codes[i]]['pos'], G.nodes[airport_codes[j]]['pos'])
            G.add_edge(airport_codes[i], airport_codes[j], distance=distance)
    
    return G

def draw_graph(G, paths=None):
    # Extract node positions based on latitude and longitude
    pos = nx.get_node_attributes(G, 'pos')
    
    node_lon = []
    node_lat = []
    node_text = []
    for node in G.nodes():
        lon, lat = pos[node]
        node_lon.append(lon)
        node_lat.append(lat)
        node_text.append(node)
    
    trace_nodes = go.Scattergeo(
        lon=node_lon,
        lat=node_lat,
        mode='markers+text',
        text=node_text,
        marker=dict(size=15, color='darkblue'),
        textposition='top center',
        name='Airports'
    )
    
    fig = go.Figure(data=[trace_nodes])
    
    if paths:
        colors = ['red', 'blue']
        names = ['Shortest Path', 'Shortest Path (excluding direct route)']
        for idx, path in enumerate(paths):
            path_edges_x = []
            path_edges_y = []
            for u, v in zip(path, path[1:]):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                path_edges_x.extend([x0, x1, None])
                path_edges_y.extend([y0, y1, None])
            
            trace_path = go.Scattergeo(
                lon=path_edges_x,
                lat=path_edges_y,
                mode='lines',
                line=dict(width=2, color=colors[idx]),
                name=names[idx]
            )
            fig.add_trace(trace_path)
    
    fig.update_layout(
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=0),
        title='Airport Network',
        geo=dict(
            scope='world',
            projection_type='mercator',
            showland=True,
            landcolor='rgb(220, 220, 220)',  # Light gray for land
            countrycolor='rgb(204, 204, 204)',
            coastlinecolor='rgb(150, 150, 150)',  # Darker gray for coastlines
            showocean=True,
            oceancolor='rgb(180, 180, 255)',  # Light blue for oceans
            showlakes=True,
            lakecolor='rgb(200, 200, 255)'  # Light blue for lakes
        )
    )
    fig.show()


def get_source_destination(airports):
    print("Available airports:")
    for idx, code in enumerate(airports.keys()):
        print(f"{idx}: {code} ({airports[code]['name']})")
    
    source_idx = int(input("Enter the source airport index: "))
    destination_idx = int(input("Enter the destination airport index: "))
    
    airport_codes = list(airports.keys())
    source = airport_codes[source_idx]
    destination = airport_codes[destination_idx]
    
    return source, destination

def compute_shortest_path_dijkstra(G, source, destination):
    def dijkstra(graph, start, end):
        queue = [(0, start, [])]
        seen = set()
        min_dist = {start: 0}
        while queue:
            (cost, node, path) = heapq.heappop(queue)
            if node in seen:
                continue
            seen.add(node)
            path = path + [node]
            if node == end:
                return path, cost
            for neighbor in graph.neighbors(node):
                if neighbor in seen:
                    continue
                prev_cost = min_dist.get(neighbor, None)
                new_cost = cost + graph[node][neighbor]['distance']
                if prev_cost is None or new_cost < prev_cost:
                    min_dist[neighbor] = new_cost
                    heapq.heappush(queue, (new_cost, neighbor, path))
        return None, float('inf')
    return dijkstra(G, source, destination)

def compute_shortest_path_astar(G, source, destination, heuristic_type):
    def astar(graph, start, end, heuristic_func):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {start: None}
        g_score = {node: float('inf') for node in graph.nodes}
        g_score[start] = 0
        f_score = {node: float('inf') for node in graph.nodes}
        f_score[start] = heuristic_func(start, end)
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end:
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                return path[::-1], g_score[end]
            for neighbor in graph.neighbors(current):
                tentative_g_score = g_score[current] + graph[current][neighbor]['distance']
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic_func(neighbor, end)
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None, float('inf')

    def heuristic(u, v):
        pos_u = G.nodes[u]['pos']
        pos_v = G.nodes[v]['pos']
        if heuristic_type == 'euclidean':
            return calculate_custom_distance(pos_u, pos_v)
        elif heuristic_type == 'manhattan':
            return abs(pos_u[0] - pos_v[0]) * 100 + abs(pos_u[1] - pos_v[1]) * 110
        elif heuristic_type == 'chebyshev':
            return max(abs(pos_u[0] - pos_v[0]) * 100, abs(pos_u[1] - pos_v[1]) * 110)
        else:
            return 0  # Default heuristic (not recommended)
    return astar(G, source, destination, heuristic)

def compute_shortest_path_bellman_ford(G, source, destination):
    def bellman_ford(graph, start, end):
        distance = {node: float('inf') for node in graph.nodes}
        distance[start] = 0
        predecessor = {node: None for node in graph.nodes}
        for _ in range(len(graph.nodes) - 1):
            for u, v, data in graph.edges(data=True):
                weight = data['distance']
                if distance[u] + weight < distance[v]:
                    distance[v] = distance[u] + weight
                    predecessor[v] = u
                if distance[v] + weight < distance[u]:
                    distance[u] = distance[v] + weight
                    predecessor[u] = v
        for u, v, data in graph.edges(data=True):
            weight = data['distance']
            if distance[u] + weight < distance[v]:
                print("Negative weight cycle detected.")
                return None, float('inf')
        path = []
        current = end
        while current:
            path.append(current)
            current = predecessor[current]
        path = path[::-1]
        return path, distance[end]

    return bellman_ford(G, source, destination)

def calculate_custom_distance(pos_u, pos_v):
    return (((pos_u[0] - pos_v[0]) * 100) ** 2 + ((pos_u[1] - pos_v[1]) * 110) ** 2) ** 0.5

def compute_all_shortest_paths(G, source, destination):
    results = {}

    # Dijkstra's algorithm
    dijkstra_path, dijkstra_distance = compute_shortest_path_dijkstra(G, source, destination)
    results["Dijkstra"] = (dijkstra_path, dijkstra_distance)
    
    # A* algorithm with different heuristics
    heuristics = ['euclidean', 'manhattan', 'chebyshev']
    astar_results = {}
    
    for heuristic in heuristics:
        path, distance = compute_shortest_path_astar(G, source, destination, heuristic)
        if path:
            astar_results[heuristic] = (path, distance)
    
    for heuristic in heuristics:
        results[f"A* ({heuristic})"] = astar_results[heuristic]
    
    # Bellman-Ford algorithm
    bellman_ford_path, bellman_ford_distance = compute_shortest_path_bellman_ford(G, source, destination)
    results["Bellman-Ford"] = (bellman_ford_path, bellman_ford_distance)

    return results

def compute_all_shortest_paths_excluding_direct_edge(G, source, destination):
    if G.has_edge(source, destination):
        original_weight = G[source][destination]['distance']
        G.remove_edge(source, destination)
        results = compute_all_shortest_paths(G, source, destination)
        G.add_edge(source, destination, distance=original_weight)
        return results
    else:
        return compute_all_shortest_paths(G, source, destination)

if __name__ == "__main__":
    # Load airport coordinates from CSV
    file_path = 'airports.csv'
    airports = load_airport_coordinates(file_path)
    G = create_graph_from_airports(airports)
    source, destination = get_source_destination(airports)
    
    # Compute shortest paths
    shortest_paths = compute_all_shortest_paths(G, source, destination)
    shortest_paths_excl_direct = compute_all_shortest_paths_excluding_direct_edge(G, source, destination)

    # Display results in tables
    results_table = [
        ["Algorithm", "Distance"],
        ["Dijkstra", shortest_paths["Dijkstra"][1]],
        ["A* (Euclidean)", shortest_paths["A* (euclidean)"][1]],
        ["A* (Manhattan)", shortest_paths["A* (manhattan)"][1]],
        ["A* (Chebyshev)", shortest_paths["A* (chebyshev)"][1]],
        ["Bellman-Ford", shortest_paths["Bellman-Ford"][1]]
    ]

    results_table_excl_direct = [
        ["Algorithm", "Distance"],
        ["Dijkstra", shortest_paths_excl_direct["Dijkstra"][1]],
        ["A* (Euclidean)", shortest_paths_excl_direct["A* (euclidean)"][1]],
        ["A* (Manhattan)", shortest_paths_excl_direct["A* (manhattan)"][1]],
        ["A* (Chebyshev)", shortest_paths_excl_direct["A* (chebyshev)"][1]],
        ["Bellman-Ford", shortest_paths_excl_direct["Bellman-Ford"][1]]
    ]

    print("\nShortest path distances for each algorithm:\n")
    print(tabulate(results_table, headers="firstrow", tablefmt="grid"))

    print("\nShortest path distances for each algorithm (excluding direct edge):\n")
    print(tabulate(results_table_excl_direct, headers="firstrow", tablefmt="grid"))
    
    # Determine the best paths
    all_paths = {
        "Dijkstra": shortest_paths["Dijkstra"],
        "A* (best heuristic)": min((shortest_paths[f"A* ({h})"] for h in ["euclidean", "manhattan", "chebyshev"]), key=lambda x: x[1]),
        "Bellman-Ford": shortest_paths["Bellman-Ford"]
    }

    all_paths_excl_direct = {
        "Dijkstra": shortest_paths_excl_direct["Dijkstra"],
        "A* (best heuristic)": min((shortest_paths_excl_direct[f"A* ({h})"] for h in ["euclidean", "manhattan", "chebyshev"]), key=lambda x: x[1]),
        "Bellman-Ford": shortest_paths_excl_direct["Bellman-Ford"]
    }

    shortest_path_algorithm = min(all_paths, key=lambda k: all_paths[k][1])
    shortest_path, shortest_distance = all_paths[shortest_path_algorithm]

    shortest_path_algorithm_excl_direct = min(all_paths_excl_direct, key=lambda k: all_paths_excl_direct[k][1])
    shortest_path_excl_direct, shortest_distance_excl_direct = all_paths_excl_direct[shortest_path_algorithm_excl_direct]

    print(f"\nThe shortest path from {source} to {destination} is found by {shortest_path_algorithm}: {shortest_path} with total distance {shortest_distance}")
    print(f"\nThe shortest path from {source} to {destination} excluding the direct edge is found by {shortest_path_algorithm_excl_direct}: {shortest_path_excl_direct} with total distance {shortest_distance_excl_direct}")
    
    draw_graph(G, [shortest_path, shortest_path_excl_direct])

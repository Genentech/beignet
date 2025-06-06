import torch
from torch import Tensor


def yen(
    graph: Tensor,
    source: Tensor,
    sink: Tensor,
    k: Tensor,
) -> Tensor:
    r"""
    Computes k shortest paths using Yen's algorithm.

    Yen's algorithm finds the k shortest simple paths from source to sink.
    This is a simplified implementation that returns the lengths of the paths.

    Parameters
    ----------
    graph : Tensor
        Sparse CSR tensor representing the weighted adjacency matrix with
        shape (num_nodes, num_nodes). Non-zero entries represent edge weights.
    source : Tensor
        Source node index (scalar tensor).
    sink : Tensor
        Sink node index (scalar tensor).
    k : Tensor
        Number of shortest paths to find (scalar tensor).

    Returns
    -------
    path_lengths : Tensor
        Lengths of the k shortest paths with shape (k,).
        If fewer than k paths exist, returns -1 for non-existent paths.
    """
    num_nodes = graph.shape[-1]
    device = graph.device
    dtype = graph.dtype

    source_idx = source.item()
    sink_idx = sink.item()
    k_val = k.item()

    def dijkstra_modified(excluded_edges=None, excluded_nodes=None):
        """Modified Dijkstra that can exclude edges and nodes."""
        distances = torch.full((num_nodes,), float("inf"), dtype=dtype, device=device)
        distances[source_idx] = 0
        visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        parent = torch.full((num_nodes,), -1, dtype=torch.long, device=device)

        if excluded_nodes is None:
            excluded_nodes = set()
        if excluded_edges is None:
            excluded_edges = set()

        crow_indices = graph.crow_indices()
        col_indices = graph.col_indices()
        values = graph.values()

        for _ in range(num_nodes):
            # Find unvisited node with minimum distance
            min_dist = float("inf")
            current = -1

            for i in range(num_nodes):
                if (
                    not visited[i]
                    and i not in excluded_nodes
                    and distances[i] < min_dist
                ):
                    min_dist = distances[i].item()
                    current = i

            if current == -1 or min_dist == float("inf"):
                break

            visited[current] = True

            if current == sink_idx:
                # Reconstruct path length
                return distances[sink_idx], parent

            # Update neighbors
            start_idx = crow_indices[current]
            end_idx = crow_indices[current + 1]

            for edge_idx in range(start_idx, end_idx):
                neighbor = col_indices[edge_idx].item()
                edge_weight = values[edge_idx]

                if (current, neighbor) in excluded_edges or neighbor in excluded_nodes:
                    continue

                new_distance = distances[current] + edge_weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    parent[neighbor] = current

        return torch.tensor(float("inf"), dtype=dtype, device=device), parent

    def get_path_nodes(parent, sink_idx):
        """Get path nodes from parent array."""
        path = []
        current = sink_idx
        while current != -1:
            path.append(current)
            current = parent[current].item() if parent[current] != -1 else -1
        return path[::-1]

    # Find first shortest path
    shortest_distance, parent = dijkstra_modified()
    if torch.isinf(shortest_distance):
        # No path exists
        return torch.full((k_val,), -1, dtype=dtype, device=device)

    # Store k shortest paths
    k_paths = []
    k_paths.append((shortest_distance.item(), get_path_nodes(parent, sink_idx)))

    # Candidate paths (distance, path)
    candidates = []

    for _ in range(1, k_val):
        if not k_paths:
            break

        last_path = k_paths[-1][1]

        # Generate candidate paths by removing edges from the last path
        for j in range(len(last_path) - 1):
            # Exclude nodes before spur node
            excluded_nodes = set(last_path[:j])

            # Exclude edges used in previous paths at this spur node
            excluded_edges = set()
            for _, prev_path in k_paths:
                if (
                    j < len(prev_path) - 1
                    and len(prev_path) > j + 1
                    and prev_path[j] == last_path[j]
                ):
                    excluded_edges.add((prev_path[j], prev_path[j + 1]))

            # Find spur path
            spur_distance, spur_parent = dijkstra_modified(
                excluded_edges, excluded_nodes
            )

            if not torch.isinf(spur_distance):
                # Construct total path
                root_distance = 0
                for k in range(j):
                    # Add edge weights for root path
                    u, v = last_path[k], last_path[k + 1]
                    start_idx = graph.crow_indices()[u]
                    end_idx = graph.crow_indices()[u + 1]

                    for edge_idx in range(start_idx, end_idx):
                        if graph.col_indices()[edge_idx].item() == v:
                            root_distance += graph.values()[edge_idx].item()
                            break

                total_distance = root_distance + spur_distance.item()
                spur_path = get_path_nodes(spur_parent, sink_idx)
                total_path = last_path[:j] + spur_path

                candidates.append((total_distance, total_path))

        if not candidates:
            break

        # Sort candidates and pick the shortest
        candidates.sort(key=lambda x: x[0])
        next_path = candidates.pop(0)
        k_paths.append(next_path)

    # Return path lengths
    result = torch.full((k_val,), -1, dtype=dtype, device=device)
    for i, (dist, _) in enumerate(k_paths):
        if i < k_val:
            result[i] = dist

    return result

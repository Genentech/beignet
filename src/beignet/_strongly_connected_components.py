import torch
from torch import Tensor


def strongly_connected_components(
    graph: Tensor,
) -> Tensor:
    r"""
    Finds strongly connected components using Tarjan's algorithm.

    A strongly connected component is a maximal set of vertices such that
    there is a path from every vertex to every other vertex in the set.

    Parameters
    ----------
    graph : Tensor
        Sparse CSR tensor representing the adjacency matrix with
        shape (num_nodes, num_nodes). Non-zero entries represent edges.

    Returns
    -------
    labels : Tensor
        Component labels for each node with shape (num_nodes,).
        Nodes in the same component have the same label.
    """
    num_nodes = graph.shape[-1]
    device = graph.device

    # Tarjan's algorithm state
    index = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    lowlink = torch.zeros(num_nodes, dtype=torch.long, device=device)
    on_stack = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    stack = torch.zeros(num_nodes, dtype=torch.long, device=device)
    stack_size = 0

    current_index = 0
    labels = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    current_component = 0

    # Get graph structure
    crow_indices = graph.crow_indices()
    col_indices = graph.col_indices()

    def tarjan_dfs(v):
        nonlocal current_index, current_component, stack_size

        # Set the depth index for v
        index[v] = current_index
        lowlink[v] = current_index
        current_index += 1

        # Push v onto stack
        stack[stack_size] = v
        stack_size += 1
        on_stack[v] = True

        # Consider successors of v
        start_idx = crow_indices[v]
        end_idx = crow_indices[v + 1]

        for edge_idx in range(start_idx, end_idx):
            w = col_indices[edge_idx].item()

            if index[w] == -1:
                # Successor w has not been visited; recurse
                tarjan_dfs(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                # Successor w is in stack S and hence in the current SCC
                lowlink[v] = min(lowlink[v], index[w])

        # If v is a root node, pop the stack and create an SCC
        if lowlink[v] == index[v]:
            # Start a new strongly connected component
            while True:
                stack_size -= 1
                w = stack[stack_size].item()
                on_stack[w] = False
                labels[w] = current_component
                if w == v:
                    break
            current_component += 1

    # Run Tarjan's algorithm on all unvisited nodes
    for v in range(num_nodes):
        if index[v] == -1:
            tarjan_dfs(v)

    return labels

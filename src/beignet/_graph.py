from torch import Tensor


def breath_first_ordering(input: Tensor, start: Tensor) -> Tensor:
    """
    Breadth-first orderings of directed graphs.

    Parameters
    ----------
    input : Tensor
        Square matrices representing directed graphs.

    start : int
        Indexes of the starting nodes.

    Returns
    -------
    output : Tensor
        Breadth-first orderings of directed graphs.

    Note
    ----
    A breadth-first ordering is not unique, but a breadth-first tree that
    yields a breath-first ordering is unique.
    """
    pass


def breadth_first_tree(input: Tensor, start: Tensor) -> Tensor:
    """
    Breadth-first trees of directed graphs.

    Parameters
    ----------
    input : Tensor
        Square matrices representing directed graphs.

    start : int
        Indexes of the starting nodes.

    Returns
    -------
    output : Tensor
        Breadth-first trees of directed graphs.
    """
    pass


def cuthill_mckee(input: Tensor) -> Tensor:
    """
    Cuthill-McKee orderings of sparse matrices.

    Parameters
    ----------
    input : Tensor
        Sparse matrices.

    Returns
    -------
    output : Tensor
        Cuthill-McKee orderings of sparse matrices.
    """
    pass


def depth_first_ordering(input: Tensor, start: Tensor) -> Tensor:
    """
    Depth-first orderings of directed graphs.

    Parameters
    ----------
    input : Tensor
        Square matrices representing directed graphs.

    start : int
        Indexes of the starting nodes.

    Note
    ----
    Neither the depth-first ordering or the depth-first tree that yields the
    depth-first ordering is unique.

    Returns
    -------
    output : Tensor
        Depth-first orderings of directed graphs.
    """
    pass


def depth_first_tree(input: Tensor, start: Tensor) -> Tensor:
    """
    Depth-first trees of directed graphs.

    Parameters
    ----------
    input : Tensor
        Square matrices representing directed graphs.

    start : int
        Indexes of the starting nodes.

    Returns
    -------
    output : Tensor
        Depth-first trees of directed graphs.
    """
    pass


def dinic(input: Tensor, source: Tensor, sink: Tensor) -> Tensor:
    """
    Maximize the flow between two nodes.

    Parameters
    ----------
    input :    Tensor
        Square matrices representing a directed graph whose :math:`(i, j)`
        entry is an integer representing the capacity of the edge between
        vertices :math:`i` and :math:`j`.

    source : Tensor
        Indexes of the source nodes.

    sink : int
        Indexes of the sink nodes.

    Returns
    -------
    output : Tensor
        Maximum flows from the source nodes to the sink nodes.
    """
    pass


def edmonds_karp(input: Tensor, source: Tensor, sink: Tensor) -> Tensor:
    """
    Maximize the flow between two nodes.

    Parameters
    ----------
    input : Tensor
        Square matrices representing a directed graph whose :math:`(i, j)`
        entry is an integer representing the capacity of the edge between
        vertices :math:`i` and :math:`j`.

    source : Tensor
        Indexes of the source nodes.

    sink : int
        Indexes of the sink nodes.

    Returns
    -------
    output : Tensor
        Maximum flows from the source nodes to the sink nodes.
    """
    pass


def hopcroft_karp(input: Tensor) -> Tensor:
    """
    Maximum cardinality matchings of bipartite graphs.

    Parameters
    ----------
    input : Tensor
        Matrices representing bipartite graphs.

    Returns
    -------
    output : Tensor
        Square matrices representing maximum cardinality matchings of bipartite graphs.
    """
    pass


def kruskal(input: Tensor) -> Tensor:
    """
    Minimum spanning forests of undirected edge-weighted graphs.

    Parameters
    ----------
    input : Tensor
        Square matrices representing undirected edge-weighted graphs.

    Returns
    -------
    output : Tensor
        Square matrices representing undirected minimum spanning forests.

    Note
    ----
    If a graph is unconnected, the minimum spanning forest is returned (i.e.,
    the union of the minimum spanning trees on each connected component).

    Example
    -------
    >>> import beignet
    >>> import torch
    >>> input = torch.tensor([[0, 8, 0, 3], [0, 0, 2, 5], [0, 0, 0, 6], [0, 0, 0, 0]])
    >>> input = input.to_sparse_csr()
    >>> output = beignet.kruskal(input)
    tensor([[0, 0, 0, 3],
            [0, 0, 2, 5],
            [0, 0, 0, 0],
            [0, 0, 0, 0]])
    """
    pass


def laplacian_matrix(input: Tensor) -> Tensor:
    """
    Laplacian matrices of directed graphs.

    Parameters
    ----------
    input : Tensor
        The adjacency matrix of the graph, where `input[i, j]` represents the
        weight of the edge from node `i` to node `j`.

    Returns
    -------
    output : Tensor
        Laplacian matrices of directed graphs.
    """
    pass


def reverse_cuthill_mckee(input: Tensor) -> Tensor:
    """
    Reverse Cuthill-McKee orderings of sparse matrices.

    Parameters
    ----------
    input : Tensor
        Sparse matrices.

    Returns
    -------
    output : Tensor
        Reverse Cuthill-McKee orderings of sparse matrices.
    """
    pass


def symmetrically_normalized_laplacian_matrix(input: Tensor) -> Tensor:
    """
    Symmetrically normalized Laplacian matrices of directed graphs.

    Parameters
    ----------
    input : Tensor
        The adjacency matrix of the graph, where `input[i, j]` represents the
        weight of the edge from node `i` to node `j`.

    Returns
    -------
    output : Tensor
        Symmetrically normalized Laplacian matrices.
    """
    pass

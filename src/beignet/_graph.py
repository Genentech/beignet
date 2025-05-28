from jaxtyping import Float, Int
from torch import Tensor


def breath_first_ordering(
    input: Float[Tensor, "batch n n"],
    start: Int[Tensor, " batch"],
) -> Float[Tensor, "batch n"]:
    """
    Parameters
    ----------
    input : Float[Tensor, "batch n n"]
        Directed graphs.

    start : Int[Tensor, "batch"]
        Start node indexes.

    Returns
    -------
    output : Float[Tensor, "batch n"]
        Breadth-first orderings.
    """
    pass


def breadth_first_tree(
    input: Float[Tensor, "batch n n"],
    start: Int[Tensor, " batch"],
) -> Float[Tensor, "batch n"]:
    """
    Parameters
    ----------
    input : Float[Tensor, "batch n n"]
        Directed graphs.

    start : Int[Tensor, "batch"]
        Start node indexes.

    Returns
    -------
    output : Float[Tensor, "batch n"]
        Breadth-first trees.
    """
    pass


def cuthill_mckee(
    input: Float[Tensor, "batch m n"],
) -> Float[Tensor, "batch m"]:
    """
    Parameters
    ----------
    input : Float[Tensor, "batch m n"]
        Sparse matrices.

    Returns
    -------
    output : Float[Tensor, "batch m"]
        Cuthill-McKee orderings
    """
    pass


def depth_first_ordering(
    input: Float[Tensor, "batch n n"],
    start: Int[Tensor, " batch"],
) -> Float[Tensor, "batch n"]:
    """
    Parameters
    ----------
    input : Float[Tensor, "batch n n"]
        Directed graphs.

    start : Int[Tensor, " batch"]
        Start node indexes.

    Returns
    -------
    output : Float[Tensor, "batch n"]
        Depth-first orderings.
    """
    pass


def depth_first_tree(
    input: Float[Tensor, "batch n n"],
    start: Int[Tensor, " batch"],
) -> Float[Tensor, "batch n"]:
    """
    Parameters
    ----------
    input : Float[Tensor, "batch n n"]
        Directed graphs.

    start : Int[Tensor, " batch"]
        Start node indexes.

    Returns
    -------
    output : Float[Tensor, "batch n"]
        Depth-first trees.
    """
    pass


def dinic(
    input: Int[Tensor, "batch n n"],
    source: Int[Tensor, " batch"],
    sink: Int[Tensor, " batch"],
) -> Float[Tensor, " batch"]:
    r"""
    Parameters
    ----------
    input : Int[Tensor, "batch n n"],
        Directed graphs where :math:`(i, j)` is the capacity of the edge
        between :math:`i` and :math:`j`.

    source : Int[Tensor, "batch"],
        Source node indexes.

    sink : Int[Tensor, "batch"],
        Sink node indexes.

    Returns
    -------
    output : Float[Tensor, "batch"]
        Maximum flows.
    """
    pass


def edmonds_karp(
    input: Int[Tensor, "batch n n"],
    source: Int[Tensor, " batch"],
    sink: Int[Tensor, " batch"],
) -> Float[Tensor, " batch"]:
    """
    Parameters
    ----------
    input : Int[Tensor, "batch n n"],
        Directed graphs where :math:`(…, i, j)` are the capacities of the edges
        between :math:`i` and :math:`j`.

    source : Int[Tensor, "batch"],
        Source node indexes.

    sink : Int[Tensor, "batch"],
        Sink node indexes.

    Returns
    -------
    output : Float[Tensor, "batch"]
        Maximum flows.
    """
    pass


def hopcroft_karp(
    input: Float[Tensor, "batch m n"],
) -> Float[Tensor, "batch m n"]:
    """
    Parameters
    ----------
    input : Float[Tensor, "batch m n"]
        Bipartite graphs.

    Returns
    -------
    output : Float[Tensor, "batch m n"]
        Maximum cardinality matchings.
    """
    pass


def kruskal(
    input: Float[Tensor, "batch n n"],
) -> Float[Tensor, "batch n n"]:
    """
    Parameters
    ----------
    input : Float[Tensor, "batch n n"]
        Undirected graphs.

    Returns
    -------
    output : Float[Tensor, "batch n n"]
        Undirected minimum spanning forests.
    """
    pass


def laplacian_matrix(
    input: Float[Tensor, "batch n n"],
) -> Float[Tensor, "batch n n"]:
    """
    Parameters
    ----------
    input : Float[Tensor, "batch n n"]
        Directed graphs.

    Returns
    -------
    output : Float[Tensor, "batch n n"]
        Laplacian matrices.
    """
    pass


def reverse_cuthill_mckee(
    input: Float[Tensor, "batch m n"],
) -> Float[Tensor, "batch m"]:
    """
    Parameters
    ----------
    input : Float[Tensor, "batch m n"]
        Sparse matrices.

    Returns
    -------
    output : Float[Tensor, "batch m"]
        Reverse Cuthill-McKee orderings
    """
    pass


def strongly_connected_components(
    input: Float[Tensor, "batch n n"],
) -> Int[Tensor, "batch n"]:
    """
    Parameters
    ----------
    input : Float[Tensor, "batch n n"]
        Directed graphs.

    Returns
    -------
    output : Int[Tensor, "batch n"]
        Strongly connected components.
    """
    pass


def symmetrically_normalized_laplacian_matrix(
    input: Float[Tensor, "batch n n"],
) -> Float[Tensor, "batch n n"]:
    """
    Parameters
    ----------
    input : Float[Tensor, "batch n n"]
        Directed graphs.

    Returns
    -------
    output : Float[Tensor, "batch n n"]
        Symmetrically normalized Laplacian matrices.
    """
    pass


def weak_components(
    input: Float[Tensor, "batch n n"],
) -> Int[Tensor, "batch n"]:
    """
    Parameters
    ----------
    input : Float[Tensor, "batch n n"]
        Directed graphs.

    Returns
    -------
    output : Int[Tensor, "batch n"]
        Weak components.
    """
    pass

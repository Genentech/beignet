from beignet import ranked_fft
import numpy as np


def test_ranked_fft():
    library = np.array(
        [
            "AAAA",
            "GGGG",
            "CCCC",
            "TTTT",
        ]
    )
    ranking_scores = np.array([3, 2, 1, 4])
    n = 2
    selected = ranked_fft(library, ranking_scores, n, descending=True)

    assert np.all(selected == np.array([3, 0]))

    selected = ranked_fft(library, ranking_scores, n, descending=False)
    assert np.all(selected == np.array([2, 1]))

    ranking_scores = None
    selected = ranked_fft(library, ranking_scores, n)
    assert np.all(selected == np.array([0, 1]))

    # harder example with wider spread of distances
    library = np.array(
        [
            "AAAA",
            "GGGG",
            "CCCC",
            "TTTT",
            "ACGT",
            "TGCA",
            "ACGT",
            "TGCA",
        ]
    )
    ranking_scores = np.array(list(range(8)))
    n = 3
    selected = ranked_fft(library, ranking_scores, n, descending=True)
    assert np.all(selected == np.array([7, 6, 3]))

    selected = ranked_fft(library, ranking_scores, n, descending=False)
    assert np.all(selected == np.array([0, 1, 2]))

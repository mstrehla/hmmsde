import numpy as np

from horenko_hmmsde.pcca import pcca_plus, select_macrostate_count, sort_eigendecomposition


def test_pcca_recovers_two_blocks_on_three_state_chain():
    # States 0 and 1 rapidly communicate; state 2 is the second metastable block.
    T = np.array([
        [0.970, 0.025, 0.005],
        [0.025, 0.970, 0.005],
        [0.005, 0.005, 0.990],
    ])
    chi, eigvals, vertices = pcca_plus(T, 2)

    # Rows are probabilities.
    np.testing.assert_allclose(chi.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(chi >= -1e-12)

    # States 0 and 1 should have essentially the same macro-membership,
    # while state 2 should belong to the opposite macrostate.
    assert np.linalg.norm(chi[0] - chi[1]) < 1e-10
    assert np.linalg.norm(chi[0] - chi[2]) > 0.9


def test_spectral_gap_selects_three_for_clear_three_block_chain():
    T = np.array([
        [0.985, 0.010, 0.005, 0.000],
        [0.010, 0.985, 0.005, 0.000],
        [0.005, 0.005, 0.985, 0.005],
        [0.000, 0.000, 0.005, 0.995],
    ])
    eigvals, _ = sort_eigendecomposition(T)
    L_hat, its = select_macrostate_count(
        eigvals, tau=0.1, min_K=2, gap_threshold=2.0, fast_floor_factor=50.0
    )
    assert 2 <= L_hat <= 4

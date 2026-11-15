import numpy as np

from mtah.augment import CorruptCfg, corrupt, vball_fill, vball_params


def _toy(T=400, S=4, seed=0):
    rng = np.random.default_rng(seed)
    pxy = rng.uniform(-40, 40, (T, S, 2)).astype(np.float32)
    bxy = rng.uniform(-40, 40, (T, 2)).astype(np.float32)
    return pxy, np.array([0, 0, 1, 1], np.int8), bxy


def test_vball_p_zero_is_the_old_corruption():
    pxy, team, bxy = _toy()
    a = corrupt(pxy, team, bxy, None, np.random.default_rng(1), CorruptCfg())
    b = corrupt(pxy, team, bxy, None, np.random.default_rng(1), CorruptCfg(vball_p=0.0))
    for x, y in zip(a[:3], b[:3]):
        np.testing.assert_array_equal(x, y)


def test_vball_fill_only_fills_gaps():
    pxy, _, clean = _toy()
    bxy = clean.copy()
    bxy[100:200] = np.nan
    out = vball_fill(bxy, clean, pxy, np.random.default_rng(0), fill=1.0, prec=1.0, err=0.0)
    np.testing.assert_array_equal(out, clean)                    # perfect detector restores it
    out = vball_fill(bxy, clean, pxy, np.random.default_rng(0), fill=0.5, prec=0.7, err=0.5, burst=12)
    np.testing.assert_array_equal(out[:100], clean[:100])        # never touches seen frames
    np.testing.assert_array_equal(out[200:], clean[200:])
    assert 0 < np.mean(~np.isnan(out[100:200, 0])) < 1


def test_vball_p_one_fills_some_gaps():
    pxy, team, bxy = _toy()
    cfg = CorruptCfg(ball_missing=0.5, flip=False, team_swap=False)
    off = corrupt(pxy, team, bxy, None, np.random.default_rng(3), cfg)[2]
    on = corrupt(pxy, team, bxy, None, np.random.default_rng(3), CorruptCfg(
        ball_missing=0.5, flip=False, team_swap=False, vball_p=1.0))[2]
    assert np.isnan(on[:, 0]).sum() < np.isnan(off[:, 0]).sum()


def test_vball_params_ranges():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = vball_params(rng)
        assert 0.3 <= p["fill"] <= 0.8 and 0.5 <= p["prec"] <= 0.95 and 0.2 <= p["err"] <= 1.0
        assert p["burst"] == 1.0 or 5.0 <= p["burst"] <= 30.0

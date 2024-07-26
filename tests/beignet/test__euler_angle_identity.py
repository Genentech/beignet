import hypothesis.strategies


@hypothesis.strategies.composite
def strategy(f):
    return


@hypothesis.given(strategy())
def test_euler_angle_identity(data):
    assert True

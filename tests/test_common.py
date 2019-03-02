from elbridge.common import bound


def test_bound():
    assert bound(-1, 0, 5) == 0
    assert bound(2, 0, 5) == 2
    assert bound(6, 0, 5) == 5

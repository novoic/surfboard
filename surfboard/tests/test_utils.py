import pytest

from surfboard.utils import numseconds_to_numsamples


@pytest.mark.parametrize("input, expected", [((0.1, 18), 2),
                                             ((102, 10), 1024),
                                             ((103, 10), 1024)])
def test_numseconds_to_numsamples(input, expected):
    assert numseconds_to_numsamples(*input) == expected

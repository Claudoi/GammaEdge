import numpy as np
import pytest

from portfolio.core.guards import box_feasible, validate_weights


def test_box_feasible():
    assert box_feasible(5, 0.0, 0.6)
    assert not box_feasible(3, 0.5, 0.6)  # 3*0.5 > 1

def test_validate_weights():
    w = np.array([0.2, 0.3, 0.5])
    validate_weights(w, 0.0, 1.0)
    with pytest.raises(AssertionError):
        validate_weights(np.array([0.2, 0.3, 0.6]), 0.0, 1.0)  # sum != 1

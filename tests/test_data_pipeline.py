import numpy as np
import pandas as pd
from portfolio.features.returns import simple_returns, to_frequency

def test_resample_simple_returns():
    # Serie sintética: 1% diario → mensual ~ (1.01)^k - 1
    idx = pd.date_range("2020-01-01", periods=22, freq="B")
    px = pd.DataFrame({"A":[100*(1.01)**i for i in range(len(idx))]}, index=idx)
    r = simple_returns(px)
    rm = to_frequency(r, "M")
    assert rm.shape[0] == 1
    approx = (1.01**(len(idx)-1)) - 1
    assert abs(float(rm.iloc[0,0]) - approx) < 1e-6

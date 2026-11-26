import numpy as np
import pandas as pd

from rh import site


def test_page_builds_with_data_and_format_parts():
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.Series(np.random.default_rng(0).normal(0, 1, 300), index=idx)
    html = site.build_html(r, "2025-12")
    assert "$" + "data" not in html and "$" + "js" not in html
    assert "Updated " in html and "Data: IIM Ahmedabad" in html and "Research only" in html
    assert '"dates":["2020-01-01"' in html and "How to read it:" in html

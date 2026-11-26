import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from rh import hmm

JS = Path(__file__).parents[1] / "src" / "rh" / "rh.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def test_js_fit_and_filter_match_python():
    rng = np.random.default_rng(4)
    s = (np.cumsum(rng.random(1500) < 0.03) % 2).astype(int)
    x = np.where(s, rng.normal(-0.2, 2.0, 1500), rng.normal(0.1, 0.8, 1500))
    py = hmm.fit(x, 2, restarts=1)
    f_py, *_ = hmm.forward(x, py["pi"], py["A"], py["mu"], py["sd"])
    code = (f"const RH = require({json.dumps(str(JS))}); const x = {json.dumps(x.tolist())};"
            "const p = RH.fit(x, 2); const f = RH.forward(x, p);"
            "console.log(JSON.stringify({p, last: Array.from(f.alpha[x.length - 1])}));")
    js = json.loads(subprocess.run(["node", "-e", code], capture_output=True, text=True,
                                   check=True).stdout)
    assert np.isclose(js["p"]["ll"], py["ll"], rtol=1e-9)
    for k in ("mu", "sd"):
        assert np.allclose(js["p"][k], py[k], rtol=1e-7)
    assert np.allclose(js["p"]["A"], py["A"], atol=1e-8)
    assert np.allclose(js["last"], f_py[-1], atol=1e-8)

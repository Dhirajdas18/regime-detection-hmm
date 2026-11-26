"""The regime lab: one static Plotly page carrying IIMA's daily market returns (rh.js fits the
HMM in the browser)."""

import datetime
import json
import re
from pathlib import Path
from string import Template

import pandas as pd

from rh import data, plots

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
ICON = (
    "data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22%3E"
    "%3Cpath d=%22M1 8h3l1-4 2 8 2-6 1 2h5%22 fill=%22none%22"
    " stroke=%22%232a78d6%22 stroke-width=%221.8%22 stroke-linejoin=%22round%22/%3E"
    "%3C/svg%3E"
)
HERE = Path(__file__).parent
JS = (HERE / "rh.js").read_text(encoding="utf-8")
LAB = (HERE / "lab.html").read_text(encoding="utf-8")
PAGE = Template((HERE / "page.html").read_text(encoding="utf-8"))
REPO = "https://github.com/Dhirajdas18/regime-detection-hmm"


def lab_data(r: pd.Series) -> dict:
    """Dates and daily log returns in % (4 decimals)."""
    return {"dates": list(r.index.strftime("%Y-%m-%d")),
            "r": [round(float(v), 4) for v in r.to_numpy()]}


def lab_section(d: dict) -> str:
    return LAB.replace("$js", JS).replace("$data", json.dumps(d, separators=(",", ":")))


def method_section(_: dict) -> str:
    return """<ul class="plain">
<li>Returns: IIMA’s daily market return as a log return in percent. Annual figures multiply daily
means by the period’s trading days a year (about 248) and daily volatility by its square
root.</li>
<li>Model: a Gaussian hidden Markov model with two or three states, each with its own mean and
volatility and a fixed chance of switching each day, fitted by expectation-maximisation
(Baum-Welch) from a start that splits days by the size of their moves. States are sorted by
volatility, so the last is always the most turbulent. The page uses one start; the findings in
the repository use several.</li>
<li>Filtered probabilities use only the days up to each date, given the fitted model; smoothed
(hindsight) probabilities use every day in the period. Both use a model fitted on the whole period
you pick, so even the filtered line knows the states’ volatilities in advance; the repository’s
finding 2 shows the truly real-time version, refitted each year on past days only.</li>
<li>Research only, not investment advice.</li>
</ul>"""


SECTIONS = [
    ("Regime lab", "Pick the years and the number of states. Research only, not investment "
     "advice.", lab_section),
    ("Method and caveats", "What the model assumes, and what the page leaves out.",
     method_section),
]


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_html(r: pd.Series, release: str) -> str:
    d = lab_data(r)
    parts, nav = [], []
    for title, blurb, make in SECTIONS:
        sid = _slug(title)
        nav.append(f'<li><a href="#{sid}">{title}</a></li>')
        parts.append(f'<section id="{sid}">\n<h2>{title}</h2>\n<p>{blurb}</p>\n{make(d)}\n'
                     "</section>")
    dk, s = plots.DARK, plots.SERIES
    return PAGE.substitute(
        updated=datetime.date.today().strftime("%d %b %Y").lstrip("0"),
        release=release, first=f"{r.index[0]:%b %Y}", last=f"{r.index[-1]:%b %Y}",
        citation=data.CITATION, icon=ICON, font=FONT, repo=REPO,
        surface=plots.SURFACE, text=plots.TEXT, text2=plots.TEXT_2, grid=plots.GRID,
        accent=s[0], accent2=s[1], green=s[2], violet=s[6], muted=plots.MUTED,
        d_surface=dk[plots.SURFACE], d_text=dk[plots.TEXT], d_text2=dk[plots.TEXT_2],
        d_grid=dk[plots.GRID], d_accent=dk[s[0]], d_accent2=dk[s[1]], d_green=dk[s[2]],
        d_violet=dk[s[6]], d_muted="#6b6a64",
        nav="\n".join(nav), sections="\n".join(parts),
    )


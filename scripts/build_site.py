"""Build site/index.html from the fetched IIMA daily data (run scripts/fetch_data.py first)."""

from pathlib import Path

from rh import data
from rh.reports import market
from rh.site import build_html


def main() -> None:
    r = market()
    out = Path("site/index.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(build_html(r, data.release()), encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e3:.0f} kB), {len(r)} days")


if __name__ == "__main__":
    main()

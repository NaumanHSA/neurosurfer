#!/usr/bin/env python3
"""Rasterize the button SVGs to transparent PNGs.

The README hotlinks PNGs (not SVGs) because PyPI won't render SVGs served from
raw.githubusercontent.com. This regenerates docs/assets/buttons/pngs/ from the
SVGs in docs/assets/buttons/, preserving a transparent background.

Usage:
    pip install cairosvg      # needs system cairo (libcairo2)
    python scripts/svg_to_png.py [--scale 2] [--clean]

--clean removes any stale PNGs in the output dir first.
"""
from __future__ import annotations

import argparse
import glob
import os

import cairosvg  # type: ignore

HERE = os.path.dirname(__file__)
SVG_DIR = os.path.normpath(os.path.join(HERE, "..", "docs", "assets", "buttons"))
PNG_DIR = os.path.join(SVG_DIR, "pngs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=2.0,
                    help="output pixel density multiplier (default 2 for crisp @2x)")
    ap.add_argument("--clean", action="store_true", help="remove existing PNGs first")
    args = ap.parse_args()

    os.makedirs(PNG_DIR, exist_ok=True)

    if args.clean:
        for stale in glob.glob(os.path.join(PNG_DIR, "*.png")):
            os.remove(stale)
            print(f"  removed {os.path.basename(stale)}")

    svgs = sorted(glob.glob(os.path.join(SVG_DIR, "*.svg")))
    if not svgs:
        raise SystemExit(f"no SVGs found in {SVG_DIR}")

    for svg in svgs:
        name = os.path.splitext(os.path.basename(svg))[0]
        out = os.path.join(PNG_DIR, f"{name}.png")
        # background_color=None keeps transparency; scale bumps the raster density.
        cairosvg.svg2png(url=svg, write_to=out, scale=args.scale, background_color=None)
        print(f"  {name}.svg -> pngs/{name}.png")

    print(f"Wrote {len(svgs)} PNGs to {PNG_DIR}")


if __name__ == "__main__":
    main()

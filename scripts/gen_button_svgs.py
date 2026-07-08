#!/usr/bin/env python3
"""Generate the monochrome README quick-link button SVGs (light + dark variants).

Writes into docs/assets/buttons/. Pure-stdlib; no dependencies.
After editing, rasterize to PNG with scripts/svg_to_png.py (the PNGs are what the
README hotlinks, since PyPI won't render raw-hosted SVGs).

Naming: `-dark`  = dark ink (black fill, light text)  → for LIGHT backgrounds (PyPI, GitHub light).
        `-light` = light ink (near-white fill, dark text) → for DARK backgrounds (GitHub dark).
"""
from __future__ import annotations

import os

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "assets", "buttons")

# label, slug, icon path (drawn in a 24x24 box, stroked with currentColor)
BUTTONS = [
    ("Quick Start", "quick-start",
     "M12 3.5c2.6 1.2 4.2 3.6 4.2 6.8 0 1.2-.3 2.3-.8 3.3l-3.4 3.4-3.4-3.4c-.5-1-.8-2.1-.8-3.3 0-3.2 1.6-5.6 4.2-6.8zM12 9.3a1.5 1.5 0 100 3 1.5 1.5 0 000-3zM9.6 16.2 8 18.5m6.4-2.3L16 18.5"),
    ("Documentation", "documentation",
     "M5 4.5h8.5a2 2 0 012 2V19.5a2.2 2.2 0 00-2-1.2H5zM19 4.5h-.5a2 2 0 00-2 2V18.3a2.2 2.2 0 012-1.2h.5zM7.5 8h6M7.5 11h6M7.5 14h4"),
    ("Examples", "examples",
     "M9 8l-4 4 4 4M15 8l4 4-4 4M13 6l-2 12"),
    ("PyPI", "pypi",
     "M12 3.5l7 3.9v7.2l-7 3.9-7-3.9V7.4zM5.3 7.6L12 11.4l6.7-3.8M12 11.4v7.6"),
    ("GitHub", "github",
     "M12 3.6a8.4 8.4 0 00-2.7 16.4c.4.1.6-.2.6-.4v-1.5c-2.3.5-2.8-1.1-2.8-1.1-.4-1-.9-1.2-.9-1.2-.8-.5.1-.5.1-.5.8.1 1.3.9 1.3.9.7 1.3 2 .9 2.5.7.1-.5.3-.9.5-1.1-1.9-.2-3.8-.9-3.8-4.1 0-.9.3-1.6.9-2.2-.1-.2-.4-1.1.1-2.2 0 0 .7-.2 2.3.8a7.9 7.9 0 014.2 0c1.6-1 2.3-.8 2.3-.8.5 1.1.2 2 .1 2.2.5.6.9 1.3.9 2.2 0 3.2-1.9 3.9-3.8 4.1.3.3.6.8.6 1.6v2.4c0 .2.2.5.6.4A8.4 8.4 0 0012 3.6z"),
    ("Discord", "discord",
     "M8.5 8.2a12 12 0 017 0c1.9.9 3 2.6 3.4 5.1.3 1.9.2 3.4.2 3.4a10 10 0 01-3 1.5l-.7-1.2a7 7 0 002-1M8.5 8.2a10 10 0 00-3 1.4M8.5 8.2 8 7m8 1.2.5-1.2M8.4 15.4a9 9 0 006.9.1M6.5 13.6c.7.9 1.9.9 2.6 0M14.9 13.6c.7.9 1.9.9 2.6 0"),
]

# Tightened geometry (less padding than the first cut).
H       = 36     # button height
FS      = 14     # label font-size
ICON    = 17     # icon box size
GAP     = 8      # icon → text gap
PAD_X   = 14     # left/right inner padding
LS      = 0.6    # letter-spacing
RX      = 8      # corner radius
CHAR_W  = FS * 0.63   # per-char advance for the (monospace) label


def make(label: str, icon: str, ink: str, fill: str, border: str) -> str:
    text_w = len(label) * (CHAR_W + LS)
    w = round(PAD_X + ICON + GAP + text_w + PAD_X)
    icon_y = (H - ICON) / 2
    text_x = PAD_X + ICON + GAP
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{H}" '
        f'viewBox="0 0 {w} {H}" role="img" aria-label="{label}">\n'
        f'  <rect x="0.75" y="0.75" width="{w - 1.5}" height="{H - 1.5}" rx="{RX}" '
        f'fill="{fill}" stroke="{border}" stroke-width="1.2"/>\n'
        f'  <g transform="translate({PAD_X},{icon_y})" fill="none" stroke="{ink}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">\n'
        f'    <path d="{icon}" transform="scale({ICON / 24})"/>\n'
        f'  </g>\n'
        f'  <text x="{text_x}" y="{H / 2}" fill="{ink}" '
        f'font-family="\'JetBrains Mono\',\'DejaVu Sans Mono\',\'SFMono-Regular\',Consolas,monospace" '
        f'font-size="{FS}" font-weight="600" letter-spacing="{LS}" '
        f'dominant-baseline="central">{label}</text>\n'
        f'</svg>\n'
    )


def main() -> None:
    out = os.path.normpath(OUT)
    os.makedirs(out, exist_ok=True)
    for label, slug, icon in BUTTONS:
        dark = make(label, icon, ink="#faf9f6", fill="#111111", border="#111111")
        light = make(label, icon, ink="#0d0d0d", fill="#f5f5f5", border="#f5f5f5")
        with open(os.path.join(out, f"{slug}-dark.svg"), "w") as f:
            f.write(dark)
        with open(os.path.join(out, f"{slug}-light.svg"), "w") as f:
            f.write(light)
        print(f"  {slug}: dark + light")
    print(f"Wrote {len(BUTTONS) * 2} SVGs to {out}")


if __name__ == "__main__":
    main()

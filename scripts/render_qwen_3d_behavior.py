#!/usr/bin/env python3
"""Render a schematic 3D visualization of the Qwen layer-24 behavior.

This is intentionally schematic, not a PCA/UMAP plot of raw activations. The
goal is to make the paper's mechanism easy to see in a README:

  - failing and passing prompts separate along a layer-24 evidence direction;
  - swapping the layer-24 state moves the edit-vs-noop score;
  - both states still sit in the same top-action region: grep.

The visual style is inspired by neural-geometry point-cloud diagrams: white
background, translucent floor, colored dots, soft shadows, and sparse labels.

Output: paper/figures/qwen_layer24_3d_behavior.svg
"""
from __future__ import annotations

import html
import math
import random
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper/figures/qwen_layer24_3d_behavior.svg"

W, H = 1600, 980
CX, CY = 770, 545
SCALE = 118

INK = "#14171a"
SEC = "#565c63"
TER = "#7b8289"
MUTE = "#9aa0a6"
EDGE = "#d9dce0"
GRID = "#edf0f2"
PLANE = "#f4f5f6"
TEAL = "#06c7a6"
TEAL_DARK = "#048a72"
GREEN = "#7fc857"
RED = "#d94b5c"
ORANGE = "#f0a33a"
MAGENTA = "#b13788"
BLUE = "#304b91"
GRAY = "#b8bdc2"

SANS = "Inter, Helvetica Neue, Helvetica, Arial, DejaVu Sans, sans-serif"
MONO = "Geist Mono, Menlo, Consolas, DejaVu Sans Mono, monospace"


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def project(x: float, y: float, z: float) -> tuple[float, float]:
    """Simple isometric-ish projection into SVG coordinates."""
    sx = CX + SCALE * (0.95 * x - 0.95 * y)
    sy = CY + SCALE * (0.42 * x + 0.42 * y - 1.04 * z)
    return sx, sy


def polygon(points: list[tuple[float, float, float]]) -> str:
    return " ".join(f"{project(x, y, z)[0]:.1f},{project(x, y, z)[1]:.1f}" for x, y, z in points)


def text(
    x: float,
    y: float,
    body: str,
    *,
    size: int = 24,
    weight: int | str = 400,
    fill: str = INK,
    anchor: str = "start",
    family: str = SANS,
    opacity: float = 1.0,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
        f'opacity="{opacity}">{esc(body)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str, width: float = 2.0,
         opacity: float = 1.0, dash: str = "", marker: str = "") -> str:
    attrs = [
        f'x1="{x1:.1f}"', f'y1="{y1:.1f}"', f'x2="{x2:.1f}"', f'y2="{y2:.1f}"',
        f'stroke="{stroke}"', f'stroke-width="{width}"', f'opacity="{opacity}"',
        'stroke-linecap="round"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    if marker:
        attrs.append(f'marker-end="url(#{marker})"')
    return f"<line {' '.join(attrs)} />"


def rounded_rect(x: float, y: float, w: float, h: float, *, fill: str = "#ffffff",
                 stroke: str = EDGE, opacity: float = 1.0, radius: float = 18) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" opacity="{opacity}" />'
    )


def circle(x: float, y: float, r: float, *, fill: str, opacity: float = 1.0,
           stroke: str = "", sw: float = 0.0) -> str:
    s = (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
        f'opacity="{opacity}"'
    )
    if stroke:
        s += f' stroke="{stroke}" stroke-width="{sw}"'
    return s + " />"


@dataclass
class Point:
    x: float
    y: float
    z: float
    color: str
    alpha: float
    r: float
    group: str

    @property
    def depth(self) -> float:
        return self.x + self.y + self.z * 0.65


def jittered_cluster(
    rng: random.Random,
    n: int,
    center: tuple[float, float, float],
    spread: tuple[float, float, float],
    colors: list[str],
    group: str,
) -> list[Point]:
    cx0, cy0, cz0 = center
    sx, sy, sz = spread
    pts: list[Point] = []
    for i in range(n):
        # A slightly curved cloud reads better than a pure Gaussian blob.
        t = rng.uniform(-1, 1)
        x = cx0 + rng.gauss(0, sx) + 0.22 * t
        y = cy0 + rng.gauss(0, sy) - 0.10 * math.sin(2.3 * t)
        z = max(0.08, cz0 + rng.gauss(0, sz) + 0.18 * math.cos(1.7 * t))
        color = colors[i % len(colors)]
        pts.append(Point(x, y, z, color, rng.uniform(0.72, 0.92), rng.uniform(3.1, 5.9), group))
    return pts


def build_svg() -> str:
    rng = random.Random(24)

    fail_center = (-1.16, -0.12, 1.08)
    pass_center = (1.03, 0.10, 1.92)

    points: list[Point] = []
    points.extend(jittered_cluster(rng, 170, fail_center, (0.38, 0.31, 0.28),
                                   [RED, ORANGE, MAGENTA, "#884c9e"], "fail"))
    points.extend(jittered_cluster(rng, 170, pass_center, (0.39, 0.33, 0.32),
                                   [TEAL, GREEN, "#35bdb3", "#9dd96b"], "pass"))
    points.extend(jittered_cluster(rng, 90, (0.0, 0.55, 0.45), (1.25, 0.40, 0.18),
                                   [GRAY, "#c9cdd1", "#d8dbde"], "context"))

    fail_xy = project(*fail_center)
    pass_xy = project(*pass_center)
    fail_floor = project(fail_center[0], fail_center[1], 0)
    pass_floor = project(pass_center[0], pass_center[1], 0)

    out: list[str] = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Schematic 3D map of Qwen layer 24 showing pass fail evidence moving while the final action remains grep.">')
    out.append("<defs>")
    out.append('<filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur in="SourceAlpha" stdDeviation="8"/><feOffset dx="0" dy="10" result="offset"/><feComponentTransfer><feFuncA type="linear" slope="0.16"/></feComponentTransfer><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>')
    out.append('<marker id="arrowTeal" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#048a72"/></marker>')
    out.append('<marker id="arrowDark" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#565c63"/></marker>')
    out.append("</defs>")
    out.append('<rect width="1600" height="980" fill="#ffffff"/>')

    # Title block.
    out.append(text(78, 94, "Qwen's layer-24 state map", size=46, weight=520))
    out.append(text(80, 136, "The test result moves the internal state. The final next action still stays grep.", size=23, fill=SEC))
    out.append(text(80, 170, "Illustrative schematic of the residual-stream result.", size=17, fill=TER))

    # Base plane and grid.
    base = [(-3.25, -2.25, 0), (3.25, -2.25, 0), (3.25, 2.25, 0), (-3.25, 2.25, 0)]
    out.append(f'<polygon points="{polygon(base)}" fill="{PLANE}" stroke="{EDGE}" stroke-width="1.2" opacity="0.88" filter="url(#softShadow)" />')
    for x in [i * 0.8 for i in range(-3, 4)]:
        a = project(x, -2.25, 0.01)
        b = project(x, 2.25, 0.01)
        out.append(line(*a, *b, stroke=GRID, width=1, opacity=0.85))
    for y in [i * 0.8 for i in range(-2, 3)]:
        a = project(-3.25, y, 0.01)
        b = project(3.25, y, 0.01)
        out.append(line(*a, *b, stroke=GRID, width=1, opacity=0.85))

    # Same-action region on the floor.
    region = [(-1.85, -0.82, 0.025), (1.72, -0.82, 0.025), (1.72, 0.78, 0.025), (-1.85, 0.78, 0.025)]
    out.append(f'<polygon points="{polygon(region)}" fill="#ffffff" stroke="#cdd2d6" stroke-width="1.2" opacity="0.66" />')
    rx, ry = project(-0.08, 0.72, 0.04)
    out.append(text(rx, ry, "same top-action region: grep", size=19, fill=SEC, anchor="middle", family=MONO, opacity=0.94))

    # Shadows.
    for p in points:
        sx, sy = project(p.x, p.y, 0.02)
        out.append(circle(sx, sy, p.r * 1.18, fill="#111111", opacity=0.075 if p.group != "context" else 0.045))

    # Vertical guide lines to floor.
    out.append(line(*fail_floor, *fail_xy, stroke=RED, width=1.6, opacity=0.35, dash="4 5"))
    out.append(line(*pass_floor, *pass_xy, stroke=TEAL_DARK, width=1.6, opacity=0.42, dash="4 5"))

    # Points, depth sorted.
    for p in sorted(points, key=lambda q: q.depth):
        px, py = project(p.x, p.y, p.z)
        out.append(circle(px, py, p.r, fill=p.color, opacity=p.alpha if p.group != "context" else 0.34))

    # Centroid markers.
    out.append(circle(*fail_xy, 12, fill=RED, opacity=0.96, stroke="#ffffff", sw=3))
    out.append(circle(*pass_xy, 12, fill=TEAL, opacity=0.96, stroke="#ffffff", sw=3))

    # Evidence direction and swap arrow.
    out.append(line(fail_xy[0] + 16, fail_xy[1] - 6, pass_xy[0] - 16, pass_xy[1] - 6,
                    stroke=TEAL_DARK, width=3.2, opacity=0.95, marker="arrowTeal"))
    mx = (fail_xy[0] + pass_xy[0]) / 2
    my = (fail_xy[1] + pass_xy[1]) / 2 + 90
    out.append(rounded_rect(mx - 185, my - 31, 370, 45, fill="#ffffff", stroke="#bdebe4", opacity=0.94, radius=16))
    out.append(text(mx, my, "pass/fail evidence direction", size=20, fill=TEAL_DARK, anchor="middle", family=MONO, weight=700))

    # Labels near clusters.
    out.append(rounded_rect(fail_xy[0] - 192, fail_xy[1] - 112, 252, 72, fill="#ffffff", stroke="#efd1d5", opacity=0.96, radius=16))
    out.append(text(fail_xy[0] - 170, fail_xy[1] - 79, "buggy prompt", size=18, fill=SEC, family=MONO, weight=700))
    out.append(text(fail_xy[0] - 170, fail_xy[1] - 49, "tests fail", size=24, fill=RED, weight=760))

    out.append(rounded_rect(pass_xy[0] + 28, pass_xy[1] - 136, 282, 72, fill="#ffffff", stroke="#bdebe4", opacity=0.96, radius=16))
    out.append(text(pass_xy[0] + 50, pass_xy[1] - 103, "fixed prompt", size=18, fill=SEC, family=MONO, weight=700))
    out.append(text(pass_xy[0] + 50, pass_xy[1] - 73, "tests pass", size=24, fill=TEAL_DARK, weight=760))

    # Swap annotation.
    sx1, sy1 = project(-1.02, 1.05, 1.12)
    sx2, sy2 = project(0.88, 1.02, 1.78)
    out.append(f'<path d="M {sx1:.1f},{sy1:.1f} C {sx1 + 140:.1f},{sy1 - 105:.1f} {sx2 - 150:.1f},{sy2 - 120:.1f} {sx2:.1f},{sy2:.1f}" fill="none" stroke="{BLUE}" stroke-width="2.5" opacity="0.58" marker-end="url(#arrowDark)" />')
    swap_label_x = sx1 + 38
    swap_label_y = max(sy1, sy2) + 96
    out.append(rounded_rect(swap_label_x - 14, swap_label_y - 24, 258, 34, fill="#ffffff", stroke="#e7eaed", opacity=0.74, radius=12))
    out.append(text(swap_label_x, swap_label_y, "layer-24 swap", size=15, fill=SEC, family=MONO, weight=700, opacity=0.92))

    # Readout card.
    card_x, card_y, card_w, card_h = 1134, 338, 430, 352
    card_right = card_x + card_w - 34
    out.append(rounded_rect(card_x, card_y, card_w, card_h, fill="#ffffff", stroke=EDGE, opacity=0.985, radius=24))
    out.append(text(card_x + 34, card_y + 52, "Action readout", size=26, weight=760))
    out.append(text(card_x + 34, card_y + 88, "The score changes.", size=17, fill=SEC))
    out.append(text(card_x + 34, card_y + 114, "The top action stays grep.", size=17, fill=SEC))

    row_y = card_y + 164
    rows = [
        ("buggy + failing tests", "+3.06", RED),
        ("fixed + passing tests", "+2.31", TEAL_DARK),
        ("failing + passing state", "+2.31", TEAL_DARK),
    ]
    out.append(text(card_x + 34, row_y - 24, "run", size=16, fill=TER, family=MONO, weight=700))
    out.append(text(card_right, row_y - 24, "edit - noop", size=16, fill=TER, anchor="end", family=MONO, weight=700))
    for i, (label, score, col) in enumerate(rows):
        y = row_y + i * 50
        out.append(line(card_x + 34, y - 32, card_right, y - 32, stroke="#eef0f2", width=1))
        out.append(text(card_x + 34, y, label, size=17, fill=SEC))
        out.append(text(card_right, y, score, size=20, fill=col, anchor="end", family=MONO, weight=760))

    out.append(line(card_x + 34, card_y + 286, card_right, card_y + 286, stroke="#eef0f2", width=1))
    out.append(text(card_x + 34, card_y + 324, "top action", size=17, fill=TER, family=MONO, weight=700))
    out.append(text(card_right, card_y + 324, "grep -> grep", size=24, fill=INK, anchor="end", family=MONO, weight=780))

    # Bottom takeaway.
    out.append(rounded_rect(88, 840, 1425, 82, fill="#fbfbfc", stroke=EDGE, opacity=1, radius=20))
    out.append(text(128, 889, "The internal evidence moves along a pass/fail direction, but it is not strong enough to leave the grep action region.", size=24, fill=INK, weight=500))

    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_svg(), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

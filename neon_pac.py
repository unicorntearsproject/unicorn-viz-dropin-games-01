"""Neon Pac — a real Pac-Man simulation rendered as a neon light-show.

Unlike a procedural "maze-ish" shader, this effect runs an **actual Pac-Man
game** on the CPU (the same architecture that makes :class:`Breakout` feel
like Breakout) and lets a GLSL shader render the true game state as glowing
neon:

* **Seven 28×31 boards** that swap at random every 15–30 s (or early when Pac
  clears one): the authentic maze, two traditional block mazes, and four
  unexpected layouts (open pillar arena, concentric rings, diagonal lattice,
  big chambers).  Walls upload to the shader as textures like Breakout's brick
  matrix; dots auto-fill every reachable cell so any layout is playable.
* A self-playing Pac-Man that navigates corridors, seeks dots, flees ghosts,
  and hunts them while powered up — drawn as the iconic chomping yellow wedge
  whose mouth points in its direction of travel.
* Four ghosts (Blinky / Pinky / Inky / Clyde) with the real arcade targeting
  AI — direct chase, ambush-ahead, flanking, and shy-Clyde — plus
  scatter/chase cycling and a frightened (blue, fleeing) mode triggered by
  power-pellets.  Rendered as the unmistakable ghost silhouette with
  directional eyes.
* A rare **alien-invasion skin** (default ~12 % per board, and can strike at
  startup): Pac becomes a blinking flying saucer and the ghosts become little
  grey aliens with big slanted almond eyes — same game, cosmetic swap only.

The whole board floats in a spinning, pulsating plasma cloud with a twinkling
starfield and streaking comets, and everything drips neon / glitter: glossy
sprites with specular highlights and sparkle glints, bloom, beat-synced pulses.

Audio reactivity (all bands wired):
  - bass    → game tempo, plasma swell, Pac/pellet swell, maze glow
  - mid     → ghost glow, palette shimmer
  - treble  → pellet + star sparkle-glitter
  - beat    → plasma heart, palette shift, brief speed surge

Randomised per activation: starting board, alien-mode roll, palette hue,
spawn cells, scatter phase, and ``self.time`` — no two runs look or play alike
from frame 0.  Config: ``board_min_s``, ``board_max_s``, ``ufo_chance``.  Live
tweakables: speed, reactivity, glitter.
"""
from __future__ import annotations

__version__ = '0.11.1'

import math

from collections import deque

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

# All boards share these dimensions so the wall/dot textures can be reused.
_W = 28
_H = 31

# Board 0 — the authentic Pac-Man board (hand-placed dots).
#   '#' wall   '.' dot   'o' power-pellet   '=' ghost-house gate (impassable)
#   ' ' open path with no pellet (ghost house interior / tunnel mouths)
_BOARD_CLASSIC = [
    "############################",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#o####.#####.##.#####.####o#",
    "#.####.#####.##.#####.####.#",
    "#..........................#",
    "#.####.##.########.##.####.#",
    "#.####.##.########.##.####.#",
    "#......##....##....##......#",
    "######.##### ## #####.######",
    "     #.##### ## #####.#     ",
    "     #.##          ##.#     ",
    "     #.## ###==### ##.#     ",
    "######.## #      # ##.######",
    "      .   #      #   .      ",
    "######.## #      # ##.######",
    "     #.## ######## ##.#     ",
    "     #.##          ##.#     ",
    "     #.## ######## ##.#     ",
    "######.## ######## ##.######",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#o..##.......##.......##..o#",
    "###.##.##.########.##.##.###",
    "###.##.##.########.##.##.###",
    "#......##....##....##......#",
    "#.##########.##.##########.#",
    "#.##########.##.##########.#",
    "#..........................#",
    "############################",
]


# --- Alternate boards, generated so dimensions/connectivity are guaranteed. ---
# Each returns an (H, W) float32 wall grid (1.0 = wall).  Dots are auto-filled
# on the reachable path cells at load time, so these only need to define walls.

def _frame() -> np.ndarray:
    """A bordered, otherwise-open board."""
    g = np.ones((_H, _W), dtype=np.float32)
    g[1:_H - 1, 1:_W - 1] = 0.0
    return g


def _gen_brick() -> np.ndarray:
    """Traditional maze feel: staggered horizontal brick walls with loops."""
    g = _frame()
    for i, r in enumerate(range(3, _H - 2, 3)):
        off = 0 if i % 2 == 0 else 4
        for c in range(2 + off, _W - 3, 8):
            g[r, c:min(c + 5, _W - 2)] = 1.0
    return g


def _gen_blocks() -> np.ndarray:
    """City-grid of rectangular blocks separated by 1–2 wide corridors."""
    g = _frame()
    for r in range(2, _H - 3, 4):
        for c in range(2, _W - 3, 5):
            g[r:r + 2, c:c + 3] = 1.0
    return g


def _gen_pillars() -> np.ndarray:
    """Wide-open arena studded with single-cell pillars — very un-Pac-Man."""
    g = _frame()
    for r in range(3, _H - 2, 3):
        for c in range(3, _W - 2, 3):
            g[r, c] = 1.0
    return g


def _gen_rings() -> np.ndarray:
    """Concentric rectangular rings joined by a central vertical corridor."""
    g = _frame()
    cmid = _W // 2
    k = 2
    while (_H - 1 - k) - k >= 3 and (_W - 1 - k) - k >= 3:
        r0, r1, c0, c1 = k, _H - 1 - k, k, _W - 1 - k
        g[r0, c0:c1 + 1] = 1.0
        g[r1, c0:c1 + 1] = 1.0
        g[r0:r1 + 1, c0] = 1.0
        g[r0:r1 + 1, c1] = 1.0
        g[r0, cmid] = 0.0                       # doorways keep every ring linked
        g[r1, cmid] = 0.0
        k += 3
    return g


def _gen_lattice() -> np.ndarray:
    """Open field textured by diagonal stripes of pillars."""
    g = _frame()
    for r in range(1, _H - 1):
        for c in range(1, _W - 1):
            if (r * 2 + c) % 5 == 0:
                g[r, c] = 1.0
    return g


def _gen_chambers() -> np.ndarray:
    """A few big rooms linked by cross corridors — spacious and unexpected."""
    g = np.ones((_H, _W), dtype=np.float32)
    for rr, cc, hh, ww in [(2, 2, 11, 10), (2, 16, 11, 10),
                           (15, 2, 13, 10), (15, 16, 13, 10), (9, 10, 8, 8)]:
        g[rr:rr + hh, cc:cc + ww] = 0.0
    g[7, 1:_W - 1] = 0.0
    g[20, 1:_W - 1] = 0.0
    for c in (6, 13, 21):
        g[1:_H - 1, c] = 0.0
    return g


# Board roster: index 0 is the authentic maze; 1–2 read traditional; 3–6 are the
# unexpected ones.  A rare UFO/alien skin can overlay any of them.
_BOARD_GENERATORS = (_gen_brick, _gen_blocks, _gen_pillars,
                     _gen_rings, _gen_lattice, _gen_chambers)

# Movement directions in (dcol, drow); row increases downward.
_UP = (0, -1)
_DOWN = (0, 1)
_LEFT = (-1, 0)
_RIGHT = (1, 0)
_DIRS = (_UP, _DOWN, _LEFT, _RIGHT)

# Ghost identity: (RGB colour, scatter-corner anchor by personality index).
_GHOST_COLORS = (
    (1.00, 0.10, 0.15),  # 0 Blinky — red
    (1.00, 0.55, 0.80),  # 1 Pinky  — pink
    (0.20, 0.90, 1.00),  # 2 Inky   — cyan
    (1.00, 0.60, 0.15),  # 3 Clyde  — orange
)


class _Ent:
    """A grid-locked mover (Pac-Man or a ghost).

    Position is the integer cell ``(col, row)`` plus a fractional ``prog`` in
    ``[0, 1)`` along the current ``(dcol, drow)`` direction toward the next
    cell.  Continuous position (used for rendering and collisions) is therefore
    ``(col + dcol * prog, row + drow * prog)``.
    """

    __slots__ = ('col', 'row', 'dcol', 'drow', 'prog', 'pers', 'home', 'mode',
                 'respawn')

    def __init__(self, col: int, row: int, pers: int = 0) -> None:
        self.col = col
        self.row = row
        self.dcol = 0
        self.drow = 0
        self.prog = 0.0
        self.pers = pers          # ghost personality index (0..3); unused for Pac
        self.home = (col, row)    # scatter-corner anchor
        self.mode = 0.0           # 0 normal, 1 frightened, 2 frightened-blink, -1 eaten
        self.respawn = 0.0        # eaten respawn countdown

    @property
    def gx(self) -> float:
        # +0.5 places the sprite in the channel centre (where dots sit), not on
        # the grid line at the cell corner.
        return self.col + 0.5 + self.dcol * self.prog

    @property
    def gy(self) -> float:
        return self.row + 0.5 + self.drow * self.prog


_VERT = """
#version 330
in vec2 in_vert;
out vec2 v_uv;
void main() {
    v_uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

_FRAG = """
#version 330
uniform float iTime;
uniform vec2  iResolution;
uniform float iOriginX;      // pixel x of grid column 0 left edge
uniform float iOriginTopY;   // pixel y (y-up) of grid row 0 top edge
uniform float iCell;         // pixel size of one square cell
uniform vec2  iGrid;         // (cols, rows)
uniform sampler2D iWall;     // r>0.5 => wall
uniform sampler2D iDot;      // 0 none, 0.5 dot, 1.0 power-pellet
uniform float iBass;
uniform float iMid;
uniform float iTreble;
uniform float iBeat;
uniform float iFlash;        // white eat/beat flash
uniform float iDeath;        // red caught flash
uniform float iHue;          // palette base
uniform float iGlitter;
uniform float iUfo;          // >0.5 => alien-invasion skin (UFO + grey aliens)
uniform vec2  iPacPos;       // grid coords
uniform vec2  iPacDir;
uniform float iPacMouth;     // 0 closed .. 1 open
uniform float iPacPower;     // frightened active glow
uniform vec2  iGhostPos[4];
uniform vec2  iGhostDir[4];
uniform vec3  iGhostCol[4];
uniform float iGhostMode[4]; // <0 hidden, 0 normal, 1 fright, 2 fright-blink
in  vec2 v_uv;
out vec4 fragColor;

#define PI  3.14159265
#define TAU 6.28318530

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }

vec3 palette(float t) {
    return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67)));
}

float wallAt(ivec2 c) {
    if (c.x < 0 || c.y < 0 || c.x >= int(iGrid.x) || c.y >= int(iGrid.y))
        return 0.0;
    return texelFetch(iWall, c, 0).r;
}

// Swirling, domain-warped plasma field in roughly [0,1].
float plasma(vec2 p, float t) {
    p *= 2.2;
    vec2 q = vec2(sin(p.y * 0.9 + t * 0.7), cos(p.x * 0.8 - t * 0.6));
    float v = sin(p.x * 1.5 + t * 1.3)
            + sin(p.y * 1.7 - t * 1.1)
            + sin((p.x + q.x) * 1.9 + (p.y + q.y) * 1.3 + t * 0.9)
            + sin(length(p - q) * 2.3 - t * 1.4);
    return v * 0.25 + 0.5;
}

// Full-screen cosmic backdrop: spinning plasma cloud + stars + comets.
vec3 backdrop(vec2 uv, float aspect, float t) {
    vec2 p = (uv - 0.5) * vec2(aspect, 1.0);
    p = rot(t * 0.08 + iBass * 0.5) * p;                 // spin
    float r = length(p);
    float pulse = 0.85 + 0.28 * sin(t * 1.5 + iBass * 3.0);
    float v = plasma(p * pulse, t);

    vec3 col = palette(v * 0.6 + iHue + t * 0.03);
    col *= col;                                          // deepen
    col *= 0.35 + 0.9 * v;                               // contrast
    // banded filaments for a churning-cloud read
    col += palette(iHue + 0.15) * pow(0.5 + 0.5 * sin(v * 9.0 + t * 1.6), 3.0) * 0.35;
    float core = smoothstep(1.25, 0.0, r) * (0.6 + iBass * 1.1 + iBeat * 0.9);
    col += palette(iHue + 0.3) * core * 0.9;             // glowing bass heart
    col *= 0.78;

    // twinkling stars
    vec2 sg = uv * vec2(aspect, 1.0) * 78.0;
    float sh = hash21(floor(sg));
    if (sh > 0.93) {
        float tw = 0.5 + 0.5 * sin(t * (2.0 + sh * 6.0) + sh * 40.0);
        float sd = length(fract(sg) - 0.5);
        col += vec3(0.8, 0.9, 1.0) * smoothstep(0.42, 0.0, sd) * tw
             * (0.6 + iTreble * 1.4);
    }

    // streaking comets with glowing tails
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        float prog = fract(t / (6.0 + fi * 2.5) + fi * 0.37);
        vec2 dir = normalize(vec2(0.85 - 0.35 * fi, -0.55 + 0.32 * fi) + 0.001);
        vec2 cp = vec2(-0.9 + fi * 0.2, 0.7 - fi * 0.35) + dir * prog * 2.4;
        vec2 rel = (uv - 0.5) * vec2(aspect, 1.0) - cp;
        float along = dot(rel, dir);
        float perp = length(rel - dir * along);
        float head = exp(-length(rel) * 40.0);
        float tail = exp(-perp * 48.0) * exp(-max(-along, 0.0) * 5.0);
        vec3 cc = mix(vec3(0.7, 0.86, 1.0), palette(iHue + fi * 0.2 + 0.5), 0.4);
        col += cc * (head * 2.4 + tail * 1.3) * (0.9 + iBeat * 0.9);
    }
    return col;
}

void main() {
    vec2 uv = v_uv;
    float aspect = iResolution.x / max(iResolution.y, 1.0);
    float t = iTime;
    vec2 frag = uv * iResolution;
    vec2 gc = vec2((frag.x - iOriginX) / iCell, (iOriginTopY - frag.y) / iCell);
    ivec2 cell = ivec2(floor(gc));
    vec2 f = fract(gc) - 0.5;

    vec3 bg = backdrop(uv, aspect, t);
    vec3 col = bg;
    bool inBoard = (gc.x >= 0.0 && gc.y >= 0.0 && gc.x < iGrid.x && gc.y < iGrid.y);

    // ---- maze walls: neon tube covering the plasma, path channels dimmed ----
    float thisWall = inBoard ? wallAt(cell) : 0.0;
    if (thisWall > 0.5) {
        col = mix(bg * 0.20, vec3(0.02, 0.05, 0.18), 0.76);   // walls 10% less opaque
        float e = 0.0;
        if (wallAt(cell + ivec2( 1, 0)) < 0.5) e = max(e, smoothstep(0.34, 0.5,  f.x));
        if (wallAt(cell + ivec2(-1, 0)) < 0.5) e = max(e, smoothstep(0.34, 0.5, -f.x));
        if (wallAt(cell + ivec2( 0, 1)) < 0.5) e = max(e, smoothstep(0.34, 0.5,  f.y));
        if (wallAt(cell + ivec2( 0,-1)) < 0.5) e = max(e, smoothstep(0.34, 0.5, -f.y));
        vec3 neon = mix(vec3(0.15, 0.5, 1.0), vec3(0.3, 0.9, 1.0),
                        0.5 + 0.5 * sin(t * 1.5 + iHue * TAU + iMid * 2.0));
        float breathe = 0.85 + 0.15 * sin(t * 2.0 + iBass * 4.0);
        col += neon * e * (1.2 + iBass * 0.7) * breathe;
        col += neon * e * e * 0.7;                       // bloom
        col += neon * 0.05 * smoothstep(0.5, 0.0, length(f));  // inner sheen
    } else if (inBoard) {
        col *= 0.46;                                     // board background 10% more opaque
        float dv = texelFetch(iDot, cell, 0).r;
        float dd = length(f);
        if (dv > 0.75) {                                 // power-pellet
            float amp = 0.5 + 0.5 * sin(t * 6.0);
            float g = smoothstep(0.27 * (0.7 + 0.3 * amp), 0.0, dd);
            vec3 pc = mix(vec3(1.0, 0.85, 0.5), palette(iHue + t * 0.2), 0.4);
            col += pc * g * (1.4 + iBass * 1.0);
            col += pc * g * g * 0.9;
            col += vec3(1.0) * smoothstep(0.08, 0.0, length(f - vec2(-0.06, 0.06))) * g;  // glint
        } else if (dv > 0.25) {                          // pellet
            float g = smoothstep(0.12, 0.0, dd);
            float spk = 1.0 + iTreble * 1.8 * hash21(vec2(cell) + floor(t * 8.0));
            col += vec3(1.0, 0.82, 0.55) * g * spk * 1.0;
        }
    }

    // ---- Pac-Man (or a UFO in alien mode) ----
    {
        vec2 pq = gc - iPacPos;
        if (iUfo > 0.5) {                                 // flying saucer
            vec2 e = vec2(pq.x, -pq.y);
            float disc = 1.0 - smoothstep(-0.03, 0.03, length(e / vec2(0.5, 0.18)) - 1.0);
            float dome = (1.0 - smoothstep(-0.03, 0.03,
                          length((e - vec2(0.0, 0.05)) / vec2(0.24, 0.26)) - 1.0))
                         * step(0.0, e.y);
            float ufo = clamp(disc + dome, 0.0, 1.0);
            vec3 metal = mix(vec3(0.30, 0.40, 0.55), vec3(0.85, 0.92, 1.0),
                             clamp(e.y * 1.6 + 0.4, 0.0, 1.0));
            vec3 domecol = mix(vec3(0.4, 1.0, 0.9), vec3(0.3, 0.7, 1.0),
                               0.5 + 0.5 * sin(t * 3.0));
            vec3 ucol = mix(metal, domecol, dome * 0.75);
            col = mix(col, ucol, ufo);
            col += vec3(1.0) * smoothstep(0.09, 0.0, length(e - vec2(-0.10, 0.13))) * ufo * 0.7;
            for (int k = -2; k <= 2; k++) {               // blinking rim lights
                vec2 lp = vec2(float(k) * 0.19, -0.015);
                vec3 lc = 0.5 + 0.5 * cos(vec3(0.0, 2.1, 4.2) + t * 6.0 + float(k) * 1.3);
                col += lc * smoothstep(0.04, 0.0, length(e - lp)) * disc * (0.7 + iBeat * 0.7);
            }
            float beam = smoothstep(0.45, 0.0, abs(e.x))
                       * smoothstep(0.0, -0.7, e.y) * (0.10 + iBass * 0.15);
            col += vec3(0.4, 1.0, 0.7) * beam;            // tractor beam
            col += ucol * smoothstep(0.95, 0.45, length(e)) * 0.12;  // aura
        } else {                                          // classic chomping wedge
            float dist = length(pq);
            float r = 0.42 + iBass * 0.06;
            vec2 dir = (length(iPacDir) > 0.01) ? normalize(iPacDir) : vec2(1.0, 0.0);
            float rel = abs(mod(atan(pq.y, pq.x) - atan(dir.y, dir.x) + PI, TAU) - PI);
            float mouthHalf = iPacMouth * 0.9;
            float body = smoothstep(0.05, -0.02, dist - r);
            float mouth = smoothstep(mouthHalf + 0.06, mouthHalf - 0.06, rel);
            float pac = body * (1.0 - mouth);
            vec3 py = mix(vec3(1.0, 0.82, 0.02), vec3(1.0, 1.0, 0.55), iPacPower);
            vec3 shaded = py * (0.55 + 0.55 * (1.0 - dist / r));       // spherical shade
            col = mix(col, shaded, pac);
            col += vec3(1.0) * smoothstep(0.13, 0.0, length(pq - vec2(-0.14, 0.15))) * pac;
            for (int k = 0; k < 2; k++) {                 // rotating sparkle glints
                float ka = t * 3.0 + float(k) * PI;
                float sd = length(pq - vec2(cos(ka), sin(ka)) * r * 0.55);
                col += vec3(1.0, 1.0, 0.9) * smoothstep(0.05, 0.0, sd) * pac;
            }
            col += py * smoothstep(r * 2.0, r * 0.7, dist) * (0.25 + iBeat * 0.55)
                 * (1.0 + 0.05 * (0.5 + 0.5 * sin(iTime * 3.0)));   // glow breathes ±5%
        }
    }

    // ---- ghosts (or grey aliens in UFO mode) ----
    for (int i = 0; i < 4; i++) {
        if (iGhostMode[i] < -0.5) continue;
        vec2 e = gc - iGhostPos[i];
        e.y = -e.y;                                       // flip so dome is up
        bool fright0 = iGhostMode[i] > 0.5;
        bool blink0  = iGhostMode[i] > 1.5;
        col += iGhostCol[i] * smoothstep(0.72, 0.4, length(e)) * 0.12;   // outer aura

        if (iUfo > 0.5) {                                 // little grey alien head
            float hw = 0.51 * (0.55 + 0.45 * clamp(e.y + 0.5, 0.0, 1.0));  // 70% wider, narrow chin
            float head = 1.0 - smoothstep(-0.03, 0.03, length(e / vec2(hw, 0.42)) - 1.0);
            if (head <= 0.0) continue;
            vec3 grey = mix(vec3(0.42, 0.52, 0.45), vec3(0.78, 0.86, 0.78),
                            clamp(e.y * 1.2 + 0.5, 0.0, 1.0));
            if (fright0) grey = mix(grey, vec3(0.4, 0.5, 1.0), blink0 ? 0.85 : 0.6);
            col = mix(col, grey, head);
            col += vec3(1.0) * smoothstep(0.09, 0.0, length(e - vec2(-0.09, 0.22))) * head * 0.6;
            for (int s = -1; s <= 1; s += 2) {            // big slanted almond eyes
                vec2 le = rot(float(s) * 0.5) * (e - vec2(float(s) * 0.16, -0.02));
                float eye = 1.0 - smoothstep(-0.02, 0.02, length(le / vec2(0.055, 0.12)) - 1.0);
                col = mix(col, vec3(0.02, 0.02, 0.04), eye * head);
                col += vec3(0.6, 1.0, 0.8)
                     * smoothstep(0.02, 0.0, length(le - vec2(0.0, 0.05))) * eye * head * 0.8;
            }
            continue;
        }

        float d = length(vec2(e.x, max(e.y, 0.0))) - 0.40;
        float skirt = -0.34 + 0.07 * abs(sin(e.x * 11.0 + t * 3.0));     // wobbling skirt
        float m = smoothstep(0.03, -0.02, d) * smoothstep(-0.02, 0.02, e.y - skirt);
        if (m <= 0.0) continue;

        bool fright = iGhostMode[i] > 0.5;
        bool blink  = iGhostMode[i] > 1.5;
        vec3 gcol = fright ? (blink ? vec3(0.95, 0.97, 1.0) : vec3(0.2, 0.25, 1.0))
                           : iGhostCol[i];
        vec3 shaded = mix(gcol * 0.65, mix(gcol, vec3(1.0), 0.55),
                          clamp(e.y * 1.4 + 0.5, 0.0, 1.0)) * (1.0 + iMid * 0.4);
        col = mix(col, shaded, m);
        col += vec3(1.0) * smoothstep(0.12, 0.0, length(e - vec2(-0.13, 0.18))) * m * 0.8;  // spec
        float gl = length(e - vec2(0.16 * sin(t * 2.0 + float(i)), 0.20));
        col += vec3(1.0) * smoothstep(0.045, 0.0, gl) * m;               // sparkle

        vec2 ed = (length(iGhostDir[i]) > 0.01)
                    ? normalize(vec2(iGhostDir[i].x, -iGhostDir[i].y)) : vec2(0.0);
        for (int s = -1; s <= 1; s += 2) {
            vec2 ec = vec2(float(s) * 0.16, 0.12);
            col += vec3(0.5, 0.7, 1.0) * smoothstep(0.17, 0.10, length(e - ec)) * m * 0.35;  // glow
            col = mix(col, vec3(0.98), smoothstep(0.13, 0.10, length(e - ec)) * m);
            if (!fright) {
                float pup = smoothstep(0.07, 0.04, length(e - ec - ed * 0.06)) * m;
                col = mix(col, vec3(0.03, 0.05, 0.30), pup);
            }
        }
        if (fright) {
            float mo = smoothstep(0.03, 0.0, abs(e.y + 0.12)) * step(abs(e.x), 0.24);
            col += vec3(0.9, 0.95, 1.0) * mo * m * 0.6;
        }
    }

    // ---- post: eat-sparkle, danger flash, glitter, vignette ----
    col += palette(iHue + 0.5) * iFlash * 0.18;
    col += vec3(1.0, 0.08, 0.08) * iDeath * 0.45;
    float sp = hash21(floor(frag * 0.5) + floor(t * 20.0));
    col += vec3(0.85, 0.92, 1.0) * step(0.994, sp) * (0.35 + iTreble * 0.9) * iGlitter;
    float vig = smoothstep(1.32, 0.18, length(uv - 0.5));
    col *= vig;
    col += palette(iHue + 0.2) * iBeat * 0.05 * (1.0 - vig);

    col = col / (1.0 + col * 0.5);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.82));
    fragColor = vec4(col, 1.0);
}
"""


class NeonPac(BaseEffect):
    """Self-playing neon Pac-Man: a real maze/ghost simulation on the CPU."""

    NAME = 'Neon Pac'
    AUTHOR = 'unicorn-viz'
    TAGS = ['games', 'arcade', 'neon', 'intense', 'hard', 'feature']
    PING_PONG_FRIENDS = [
        'Breakout', 'Galaga',
        'Laser Tunnel', 'Reactor Breach', 'Threat Matrix', 'Cyber War',
        'Audio Spectrum', 'Audio Waveforms',
    ]

    # Base movement speeds in cells/second (scaled live by tempo + params).
    _PAC_SPEED = 5.4
    _GHOST_SPEED = 4.7
    _FRIGHT_SPEED = 3.0
    _FRIGHT_TIME = 6.5
    _SCATTER_PERIOD = 7.0
    _CHASE_PERIOD = 20.0

    def _init(self) -> None:
        """Build GL resources, parse the maze, and spawn the simulation."""
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.3)),
            'glitter': float(self.config.get('glitter', 1.0)),
        }

        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._W = _W
        self._H = _H
        self._board_min = float(self.config.get('board_min_s', 15.0))
        self._board_max = float(self.config.get('board_max_s', 30.0))
        self._ufo_chance = float(self.config.get('ufo_chance', 0.12))

        # Build the 7-board roster: classic (hand-placed dots) + generated.
        self._boards: list[dict] = []
        cw = np.zeros((_H, _W), dtype=np.float32)
        cd = np.zeros((_H, _W), dtype=np.float32)
        for r, line in enumerate(_BOARD_CLASSIC):
            for c, ch in enumerate(line):
                if ch in ('#', '='):
                    cw[r, c] = 1.0
                elif ch == '.':
                    cd[r, c] = 0.5
                elif ch == 'o':
                    cd[r, c] = 1.0
        self._boards.append({'wall': cw, 'dots': cd})
        for gen in _BOARD_GENERATORS:
            self._boards.append({'wall': gen(), 'dots': None})
        for b in self._boards:
            assert b['wall'].shape == (_H, _W)

        self._wall_tex = self.ctx.texture((_W, _H), 1, dtype='f4')
        self._wall_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._dot_tex = self.ctx.texture((_W, _H), 1, dtype='f4')
        self._dot_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._dot_dirty = True

        # --- audio + game state (must precede first board load) --------------
        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._flash = 0.0
        self._death = 0.0
        self._chomp = float(self.rng.uniform(0.0, math.tau))
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._fright = 0.0
        self._mode_timer = 0.0
        self._scatter = True
        self._level_timer = 0.0
        self._freeze = 0.0
        self._ufo = 0.0
        self._board_timer = float(self.rng.uniform(self._board_min, self._board_max))

        # Placeholders so _load_board can populate them.
        self._wall = self._boards[0]['wall']
        self._passable = self._wall < 0.5
        self._reachable: set[tuple[int, int]] = set()
        self._pac = _Ent(1, 1)
        self._ghosts: list[_Ent] = []
        self._dots = np.zeros((_H, _W), dtype=np.float32)
        self._dot_total = 1.0

        self._board_idx = int(self.rng.integers(0, len(self._boards)))
        self._load_board(self._board_idx)

    # ---------------------------------------------------------------- boards
    def _bfs(self, c0: int, r0: int) -> set[tuple[int, int]]:
        """All path cells reachable from ``(c0, r0)`` (tunnel wrap included)."""
        seen = {(c0, r0)}
        q = deque([(c0, r0)])
        while q:
            c, r = q.popleft()
            for dc, dr in _DIRS:
                nc, nr = self._wrap(c + dc, r + dr)
                if self._passable[nr, nc] and (nc, nr) not in seen:
                    seen.add((nc, nr))
                    q.append((nc, nr))
        return seen

    @staticmethod
    def _nearest_in(cells: set[tuple[int, int]], tc: int, tr: int) -> tuple[int, int]:
        """Cell in ``cells`` closest to the target column/row."""
        return min(cells, key=lambda cr: (cr[0] - tc) ** 2 + (cr[1] - tr) ** 2)

    def _load_board(self, idx: int) -> None:
        """Switch to board ``idx``: upload walls, fill dots, respawn everyone."""
        board = self._boards[idx]
        self._wall = board['wall']
        self._passable = self._wall < 0.5
        self._wall_tex.write(self._wall.tobytes())

        pac_c, pac_r = self._nearest_passable(_W // 2, _H - 6)
        reach = self._bfs(pac_c, pac_r)
        self._reachable = reach

        dots = np.zeros((_H, _W), dtype=np.float32)
        if board['dots'] is not None:                    # authored dots (classic)
            explicit = board['dots']
            for c, r in reach:
                dots[r, c] = explicit[r, c]
        else:                                            # auto-fill reachable cells
            for c, r in reach:
                dots[r, c] = 0.5
            for corner in [(1, 1), (_W - 2, 1), (1, _H - 2), (_W - 2, _H - 2)]:
                pc, pr = self._nearest_in(reach, *corner)
                dots[pr, pc] = 1.0
        dots[pac_r, pac_c] = 0.0                         # clear Pac's start cell
        self._dots = dots
        self._dot_total = max(1.0, float((dots > 0.0).sum()))
        self._dot_dirty = True

        homes = [(_W - 2, 1), (1, 1), (_W - 2, _H - 2), (1, _H - 2)]
        self._pac = _Ent(pac_c, pac_r)
        self._ghosts = []
        anchors = [(2, 2), (_W - 3, 2), (2, _H - 3), (_W - 3, _H - 3)]
        self.rng.shuffle(anchors)
        for pers in range(4):
            gc, gr = self._nearest_in(reach, *anchors[pers])
            g = _Ent(gc, gr, pers)
            g.home = homes[pers]
            self._ghosts.append(g)

        self._ufo = 1.0 if self.rng.random() < self._ufo_chance else 0.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._fright = 0.0
        self._pick_dir(self._pac, ghost=False)
        for g in self._ghosts:
            self._pick_dir(g, ghost=True)

    def _next_board(self) -> None:
        """Rotate to a different random board and reset the dwell timers."""
        choices = [i for i in range(len(self._boards)) if i != self._board_idx]
        self._board_idx = int(self.rng.choice(choices))
        self._load_board(self._board_idx)
        self._board_timer = float(self.rng.uniform(self._board_min, self._board_max))
        self._level_timer = 0.0
        self._flash = max(self._flash, 0.85)

    # ------------------------------------------------------------------ maze
    def _nearest_passable(self, tc: int, tr: int) -> tuple[int, int]:
        """Return the passable cell closest to the target column/row."""
        best = (1, 1)
        bd = 1e18
        for r in range(self._H):
            for c in range(self._W):
                if self._passable[r, c]:
                    dd = (c - tc) ** 2 + (r - tr) ** 2
                    if dd < bd:
                        bd = dd
                        best = (c, r)
        return best

    def _wrap(self, col: int, row: int) -> tuple[int, int]:
        """Wrap a cell across the tunnel edges."""
        if col < 0:
            col = self._W - 1
        elif col >= self._W:
            col = 0
        if row < 0:
            row = self._H - 1
        elif row >= self._H:
            row = 0
        return col, row

    def _passable_at(self, col: int, row: int) -> bool:
        col, row = self._wrap(col, row)
        return bool(self._passable[row, col])

    # --------------------------------------------------------------- AI/dirs
    def _ghost_target(self, g: _Ent) -> tuple[float, float]:
        """Classic per-personality target tile (scatter overrides to corner)."""
        pac = self._pac
        if self._scatter:
            return g.home
        if g.pers == 0:                                  # Blinky: direct chase
            return (pac.col, pac.row)
        if g.pers == 1:                                  # Pinky: 4 ahead
            return (pac.col + pac.dcol * 4, pac.row + pac.drow * 4)
        if g.pers == 2:                                  # Inky: flank via Blinky
            ax, ay = pac.col + pac.dcol * 2, pac.row + pac.drow * 2
            blinky = self._ghosts[0]
            return (2 * ax - blinky.col, 2 * ay - blinky.row)
        d2 = (g.col - pac.col) ** 2 + (g.row - pac.row) ** 2   # Clyde: shy
        return (pac.col, pac.row) if d2 > 64 else g.home

    def _pick_dir(self, ent: _Ent, ghost: bool) -> None:
        """Choose the next direction for an entity at a decision cell."""
        rev = (-ent.dcol, -ent.drow)
        opts = [d for d in _DIRS
                if d != rev and self._passable_at(ent.col + d[0], ent.row + d[1])]
        if not opts:                                     # dead end: allow reverse
            opts = [d for d in _DIRS
                    if self._passable_at(ent.col + d[0], ent.row + d[1])]
        if not opts:
            ent.dcol = ent.drow = 0
            return

        if ghost:
            if ent.mode >= 0.5:                          # frightened: random flee
                choice = opts[int(self.rng.integers(0, len(opts)))]
            else:
                tx, ty = self._ghost_target(ent)
                choice = min(opts, key=lambda d: (ent.col + d[0] - tx) ** 2
                             + (ent.row + d[1] - ty) ** 2)
        else:
            choice = self._pick_pac_dir(ent, opts)
        ent.dcol, ent.drow = choice

    def _pick_pac_dir(self, pac: _Ent, opts: list[tuple[int, int]]) -> tuple[int, int]:
        """Score Pac's options: seek dots, dodge ghosts, hunt when powered."""
        norm_ghosts = [g for g in self._ghosts if g.mode == 0.0]
        fright_ghosts = [g for g in self._ghosts if g.mode >= 0.5]

        best = opts[0]
        best_score = -1e18
        for d in opts:
            nc, nr = self._wrap(pac.col + d[0], pac.row + d[1])
            score = 0.0
            # reward dots up to 3 cells ahead along this direction
            cc, cr = nc, nr
            w = 1.0
            for _ in range(3):
                score += float(self._dots[cr, cc]) * w
                w *= 0.6
                cc, cr = self._wrap(cc + d[0], cr + d[1])
                if not self._passable[cr, cc]:
                    break
            # flee nearest normal ghost within range
            if norm_ghosts:
                gg = min(norm_ghosts,
                         key=lambda g: (g.col - pac.col) ** 2 + (g.row - pac.row) ** 2)
                dn = math.hypot(pac.col - gg.col, pac.row - gg.row)
                if dn < 6.0:
                    dnx = math.hypot(nc - gg.col, nr - gg.row)
                    score += (dnx - dn) * 1.6
            # hunt nearest frightened ghost
            if fright_ghosts:
                gg = min(fright_ghosts,
                         key=lambda g: (g.col - pac.col) ** 2 + (g.row - pac.row) ** 2)
                dn = math.hypot(pac.col - gg.col, pac.row - gg.row)
                dnx = math.hypot(nc - gg.col, nr - gg.row)
                score += (dn - dnx) * 1.3
            score += float(self.rng.uniform(0.0, 0.4))
            if score > best_score:
                best_score = score
                best = d
        return best

    # ------------------------------------------------------------- movement
    def _advance(self, ent: _Ent, speed: float, dt: float, is_pac: bool) -> None:
        """Move an entity, choosing directions and (for Pac) eating dots."""
        if ent.dcol == 0 and ent.drow == 0:
            self._pick_dir(ent, ghost=not is_pac)
            if ent.dcol == 0 and ent.drow == 0:
                return
        ent.prog += speed * dt
        guard = 0
        while ent.prog >= 1.0 and guard < 6:
            guard += 1
            ent.prog -= 1.0
            ent.col, ent.row = self._wrap(ent.col + ent.dcol, ent.row + ent.drow)
            if is_pac:
                self._eat(ent.col, ent.row)
            self._pick_dir(ent, ghost=not is_pac)
            if ent.dcol == 0 and ent.drow == 0:
                ent.prog = 0.0
                break

    def _eat(self, col: int, row: int) -> None:
        """Consume a dot / power-pellet at Pac's cell."""
        v = self._dots[row, col]
        if v <= 0.0:
            return
        self._dots[row, col] = 0.0
        self._dot_dirty = True
        if v > 0.75:                                     # power-pellet
            self._fright = self._FRIGHT_TIME
            self._flash = max(self._flash, 0.8)
            for g in self._ghosts:
                if g.mode >= 0.0:
                    g.mode = 1.0
                    g.dcol, g.drow = -g.dcol, -g.drow    # reverse on fright
        else:
            self._flash = max(self._flash, 0.25)

    def _collisions(self) -> None:
        """Resolve Pac↔ghost contact (eat frightened, die to normal)."""
        pac = self._pac
        for g in self._ghosts:
            if g.mode < 0.0:
                continue
            if math.hypot(pac.gx - g.gx, pac.gy - g.gy) < 0.55:
                if g.mode >= 0.5:                        # eat the ghost
                    g.mode = -1.0
                    g.respawn = 1.2
                    self._flash = max(self._flash, 1.0)
                else:                                    # caught: brief reset
                    self._death = 1.0
                    self._freeze = 0.7
                    self._reset_positions()
                    return

    def _reset_positions(self) -> None:
        """Send Pac and ghosts back to spawn on the current board (after a catch)."""
        reach = self._reachable
        self._pac.col, self._pac.row = self._nearest_in(reach, self._W // 2, self._H - 6)
        self._pac.prog = 0.0
        self._pac.dcol = self._pac.drow = 0
        anchors = [(2, 2), (self._W - 3, 2), (2, self._H - 3), (self._W - 3, self._H - 3)]
        self.rng.shuffle(anchors)
        for g, a in zip(self._ghosts, anchors):
            g.col, g.row = self._nearest_in(reach, *a)
            g.prog = 0.0
            g.dcol = g.drow = 0
            g.mode = 0.0
            g.respawn = 0.0
        self._fright = 0.0
        self._pick_dir(self._pac, ghost=False)
        for g in self._ghosts:
            self._pick_dir(g, ghost=True)

    # ---------------------------------------------------------------- update
    def update(self, dt: float, audio: AudioData) -> None:
        """Advance audio envelopes and the whole game simulation."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 2.0))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 2.0))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 2.0))

        if audio.beat > 0.5:
            self._beat = 1.0
            self._flash = max(self._flash, 0.4)
            self._hue = float((self._hue + 0.02 + self._mid * 0.02) % 1.0)
        self._beat = max(0.0, self._beat - dt * 2.0)
        self._flash = max(0.0, self._flash - dt * 1.8)
        self._death = max(0.0, self._death - dt * 1.6)

        # scatter/chase cycling
        self._mode_timer += dt
        period = self._SCATTER_PERIOD if self._scatter else self._CHASE_PERIOD
        if self._mode_timer >= period:
            self._mode_timer = 0.0
            self._scatter = not self._scatter

        # frightened countdown + blink flag
        if self._fright > 0.0:
            self._fright = max(0.0, self._fright - dt)
            blink = self._fright < 1.6 and (int(self.time * 8.0) % 2 == 0)
            for g in self._ghosts:
                if g.mode >= 0.5:
                    g.mode = 2.0 if blink else 1.0
            if self._fright == 0.0:
                for g in self._ghosts:
                    if g.mode >= 0.5:
                        g.mode = 0.0

        # eaten ghost respawn (reappear near the board centre)
        for g in self._ghosts:
            if g.mode < 0.0:
                g.respawn -= dt
                if g.respawn <= 0.0:
                    g.col, g.row = self._nearest_in(self._reachable,
                                                    self._W // 2, self._H // 2)
                    g.prog = 0.0
                    g.dcol = g.drow = 0
                    g.mode = 0.0

        # chomp animation
        tempo = max(0.05, float(self.parameters['speed'])) * (1.0 + self._bass * 0.5)
        self._chomp += dt * (7.0 + self._bass * 5.0)

        # freeze after being caught
        if self._freeze > 0.0:
            self._freeze = max(0.0, self._freeze - dt)
            return

        pac_speed = self._PAC_SPEED * tempo
        self._advance(self._pac, pac_speed, dt, is_pac=True)
        for g in self._ghosts:
            if g.mode < 0.0:
                continue
            gs = (self._FRIGHT_SPEED if g.mode >= 0.5 else self._GHOST_SPEED) * tempo
            self._advance(g, gs, dt, is_pac=False)

        self._collisions()

        # rotate boards on the dwell timer, or early if Pac clears the board
        self._board_timer -= dt
        self._level_timer += dt
        remaining = float((self._dots > 0.0).sum())
        if self._board_timer <= 0.0 or remaining < 0.04 * self._dot_total:
            self._next_board()

    # ---------------------------------------------------------------- render
    def render(self) -> None:
        """Upload game state and draw the neon board."""
        if self._dot_dirty:
            self._dot_tex.write(self._dots.tobytes())
            self._dot_dirty = False

        w, h = float(self.width), float(self.height)
        cell = min(w / (self._W + 1.5), h / (self._H + 1.5))
        origin_x = (w - self._W * cell) * 0.5
        origin_top_y = (h + self._H * cell) * 0.5

        self._wall_tex.use(location=0)
        self._dot_tex.use(location=1)

        p = self._prog
        p['iTime'].value = float(self.time)
        p['iResolution'].value = (w, h)
        p['iOriginX'].value = float(origin_x)
        p['iOriginTopY'].value = float(origin_top_y)
        p['iCell'].value = float(cell)
        p['iGrid'].value = (float(self._W), float(self._H))
        p['iWall'].value = 0
        p['iDot'].value = 1
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iFlash'].value = self._flash
        p['iDeath'].value = self._death
        p['iHue'].value = self._hue
        p['iGlitter'].value = max(0.0, float(self.parameters['glitter']))
        p['iUfo'].value = self._ufo

        pac = self._pac
        pdir = (float(pac.dcol), float(pac.drow))
        if pdir == (0.0, 0.0):
            pdir = (1.0, 0.0)
        p['iPacPos'].value = (pac.gx, pac.gy)
        p['iPacDir'].value = pdir
        p['iPacMouth'].value = 0.5 + 0.5 * math.sin(self._chomp)
        p['iPacPower'].value = 1.0 if self._fright > 0.0 else 0.0

        # moderngl array uniforms take one iterable per element.
        pos: list[tuple[float, float]] = []
        dirs: list[tuple[float, float]] = []
        cols: list[tuple[float, float, float]] = []
        modes: list[float] = []
        for g in self._ghosts:
            pos.append((g.gx, g.gy))
            gd = (float(g.dcol), float(g.drow))
            if gd == (0.0, 0.0):
                gd = (1.0, 0.0)
            dirs.append(gd)
            cols.append(_GHOST_COLORS[g.pers])
            modes.append(float(g.mode))
        p['iGhostPos'].value = pos
        p['iGhostDir'].value = dirs
        p['iGhostCol'].value = cols
        p['iGhostMode'].value = modes

        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release all GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
        self._wall_tex.release()
        self._dot_tex.release()

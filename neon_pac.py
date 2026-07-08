"""Neon Pac — a real Pac-Man simulation rendered as a neon light-show.

Unlike a procedural "maze-ish" shader, this effect runs an **actual Pac-Man
game** on the CPU (the same architecture that makes :class:`Breakout` feel
like Breakout) and lets a GLSL shader render the true game state as glowing
neon:

* An authentic 28×31 maze (walls / dots / power-pellets / tunnels) stored as a
  grid and uploaded to the shader as textures — exactly like Breakout's brick
  matrix.
* A self-playing Pac-Man that navigates corridors, seeks dots, flees ghosts,
  and hunts them while powered up — drawn as the iconic chomping yellow wedge
  whose mouth points in its direction of travel.
* Four ghosts (Blinky / Pinky / Inky / Clyde) with the real arcade targeting
  AI — direct chase, ambush-ahead, flanking, and shy-Clyde — plus
  scatter/chase cycling and a frightened (blue, fleeing) mode triggered by
  power-pellets.  Rendered as the unmistakable ghost silhouette with
  directional eyes.

Everything is dripping in neon / glitter: blue outlined maze conduits, bloom,
beat-synced flashes, treble sparkle-glitter on the pellets, and chromatic
vignette pulses.

Audio reactivity (all bands wired):
  - bass    → game tempo, Pac/pellet swell, maze glow
  - mid     → ghost glow, palette shimmer
  - treble  → pellet sparkle-glitter
  - beat    → global flash, palette shift, brief speed surge

Randomised per activation: palette base hue, Pac + ghost spawn cells (nearest
passable to randomised anchors), scatter phase, and ``self.time`` — no two
runs look or play alike from frame 0.  Live tweakables: speed, reactivity,
glitter.
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

# Authentic Pac-Man board: 28 columns × 31 rows.
#   '#' wall   '.' dot   'o' power-pellet   '=' ghost-house gate (impassable)
#   ' ' open path with no pellet (ghost house interior / tunnel mouths)
_MAZE = [
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
        return self.col + self.dcol * self.prog

    @property
    def gy(self) -> float:
        return self.row + self.drow * self.prog


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

float wallAt(ivec2 c) {
    if (c.x < 0 || c.y < 0 || c.x >= int(iGrid.x) || c.y >= int(iGrid.y))
        return 0.0;
    return texelFetch(iWall, c, 0).r;
}

void main() {
    vec2 frag = v_uv * iResolution;
    vec2 gc = vec2((frag.x - iOriginX) / iCell,
                   (iOriginTopY - frag.y) / iCell);
    ivec2 cell = ivec2(floor(gc));
    vec2 f = fract(gc) - 0.5;

    vec3 col = vec3(0.010, 0.010, 0.030);
    bool inBoard = (gc.x >= 0.0 && gc.y >= 0.0 && gc.x < iGrid.x && gc.y < iGrid.y);

    // ---- maze walls: dark blue fill + bright neon edges facing open path ----
    float thisWall = inBoard ? wallAt(cell) : 0.0;
    if (thisWall > 0.5) {
        col += vec3(0.00, 0.03, 0.10);
        float e = 0.0;
        if (wallAt(cell + ivec2( 1, 0)) < 0.5) e = max(e, smoothstep(0.34, 0.5,  f.x));
        if (wallAt(cell + ivec2(-1, 0)) < 0.5) e = max(e, smoothstep(0.34, 0.5, -f.x));
        if (wallAt(cell + ivec2( 0, 1)) < 0.5) e = max(e, smoothstep(0.34, 0.5,  f.y));
        if (wallAt(cell + ivec2( 0,-1)) < 0.5) e = max(e, smoothstep(0.34, 0.5, -f.y));
        vec3 neon = mix(vec3(0.10, 0.45, 1.0), vec3(0.25, 0.85, 1.0),
                        0.5 + 0.5 * sin(iTime * 1.5 + iHue * TAU + iMid * 2.0));
        col += neon * e * (1.1 + iBass * 0.6);
        col += neon * e * e * 0.6;                 // bloom
    } else if (inBoard) {
        // ---- dots + power-pellets ----
        float dv = texelFetch(iDot, cell, 0).r;
        float dd = length(f);
        if (dv > 0.75) {                            // power-pellet: big, blinking
            float amp = 0.5 + 0.5 * sin(iTime * 6.0);
            float g = smoothstep(0.26 * (0.7 + 0.3 * amp), 0.0, dd);
            col += vec3(1.0, 0.85, 0.5) * g * (1.3 + iBass * 0.9);
            col += vec3(1.0, 0.70, 0.3) * g * g * 0.8;
        } else if (dv > 0.25) {                     // pellet: small, treble sparkle
            float g = smoothstep(0.11, 0.0, dd);
            float spk = 1.0 + iTreble * 1.6 * hash21(vec2(cell) + floor(iTime * 8.0));
            col += vec3(1.0, 0.82, 0.55) * g * spk * 0.9;
        }
    }

    // ---- Pac-Man: chomping wedge facing its travel direction ----
    {
        vec2 pq = gc - iPacPos;
        float dist = length(pq);
        float r = 0.42 + iBass * 0.06;
        vec2 dir = (length(iPacDir) > 0.01) ? normalize(iPacDir) : vec2(1.0, 0.0);
        float rel = abs(mod(atan(pq.y, pq.x) - atan(dir.y, dir.x) + PI, TAU) - PI);
        float mouthHalf = iPacMouth * 0.9;
        float body = smoothstep(0.05, -0.02, dist - r);
        float mouth = smoothstep(mouthHalf + 0.06, mouthHalf - 0.06, rel);
        float pac = body * (1.0 - mouth);
        vec3 py = mix(vec3(1.0, 0.85, 0.05), vec3(1.0, 1.0, 0.5), iPacPower);
        col += py * pac * (1.25 + iBeat * 0.4);
        col += py * pac * 0.5 * smoothstep(r * 1.9, 0.0, dist);   // halo bloom
    }

    // ---- ghosts: dome + wavy skirt silhouette with directional eyes ----
    for (int i = 0; i < 4; i++) {
        if (iGhostMode[i] < -0.5) continue;
        vec2 e = gc - iGhostPos[i];
        e.y = -e.y;                                 // flip so dome is up
        float d = length(vec2(e.x, max(e.y, 0.0))) - 0.40;
        float skirt = -0.34 + 0.07 * abs(sin(e.x * 11.0));
        float m = smoothstep(0.03, -0.02, d) * smoothstep(-0.02, 0.02, e.y - skirt);
        if (m <= 0.0) continue;

        bool fright = iGhostMode[i] > 0.5;
        bool blink  = iGhostMode[i] > 1.5;
        vec3 gcol = fright ? (blink ? vec3(0.95, 0.97, 1.0) : vec3(0.15, 0.20, 1.0))
                           : iGhostCol[i];
        col += gcol * m * (1.05 + iMid * 0.5);
        col += gcol * m * 0.5 * smoothstep(0.75, 0.0, length(e));   // bloom

        vec2 ed = (length(iGhostDir[i]) > 0.01)
                    ? normalize(vec2(iGhostDir[i].x, -iGhostDir[i].y)) : vec2(0.0);
        for (int s = -1; s <= 1; s += 2) {
            vec2 ec = vec2(float(s) * 0.16, 0.12);
            float white = smoothstep(0.13, 0.10, length(e - ec)) * m;
            col = mix(col, vec3(0.95, 0.97, 1.0), white);
            if (!fright) {
                float pup = smoothstep(0.07, 0.04, length(e - ec - ed * 0.06)) * m;
                col = mix(col, vec3(0.04, 0.05, 0.25), pup);
            }
        }
        if (fright) {                               // scared zig-zag mouth
            float mo = smoothstep(0.03, 0.0, abs(e.y + 0.12)) * step(abs(e.x), 0.24);
            col += vec3(0.9, 0.95, 1.0) * mo * m * 0.6;
        }
    }

    // ---- neon post: flashes, glitter, vignette ----
    col += vec3(1.0) * iFlash * 0.22;
    col += vec3(1.0, 0.10, 0.10) * iDeath * 0.40;
    float sp = hash21(floor(frag * 0.5) + floor(iTime * 20.0));
    col += vec3(0.8, 0.9, 1.0) * step(0.995, sp) * (0.3 + iTreble * 0.8) * iGlitter;
    float vig = smoothstep(1.30, 0.20, length(v_uv - 0.5));
    col *= vig;
    col += vec3(0.4, 0.2, 0.8) * iBeat * 0.06 * (1.0 - vig);

    col = col / (1.0 + col * 0.5);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.85));
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

        # --- parse maze into wall + dot grids -------------------------------
        self._H = len(_MAZE)
        self._W = len(_MAZE[0])
        wall = np.zeros((self._H, self._W), dtype=np.float32)
        dots0 = np.zeros((self._H, self._W), dtype=np.float32)
        passable = np.zeros((self._H, self._W), dtype=bool)
        for r, line in enumerate(_MAZE):
            for c, ch in enumerate(line):
                if ch == '#' or ch == '=':
                    wall[r, c] = 1.0
                else:
                    passable[r, c] = True
                    if ch == '.':
                        dots0[r, c] = 0.5
                    elif ch == 'o':
                        dots0[r, c] = 1.0
        self._wall = wall
        self._passable = passable
        self._dots0 = dots0
        self._dots = dots0.copy()
        self._dot_total = float((dots0 > 0.0).sum())

        self._wall_tex = self.ctx.texture((self._W, self._H), 1, dtype='f4')
        self._wall_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._wall_tex.write(wall.tobytes())
        self._dot_tex = self.ctx.texture((self._W, self._H), 1, dtype='f4')
        self._dot_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._dot_dirty = True

        # --- spawn entities at randomised (but valid) cells -----------------
        jx = int(self.rng.integers(-2, 3))
        self._pac = _Ent(*self._nearest_passable(self._W // 2 + jx, self._H - 6))
        anchors = [(2, 2), (self._W - 3, 2), (2, self._H - 3), (self._W - 3, self._H - 3)]
        self.rng.shuffle(anchors)
        homes = [(self._W - 2, 1), (1, 1), (self._W - 2, self._H - 2), (1, self._H - 2)]
        self._ghosts: list[_Ent] = []
        for pers in range(4):
            c, r = self._nearest_passable(*anchors[pers])
            g = _Ent(c, r, pers)
            g.home = homes[pers]
            self._ghosts.append(g)

        # --- audio + game state (must precede first direction pick) ----------
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

        self._pick_dir(self._pac, ghost=False)
        for g in self._ghosts:
            self._pick_dir(g, ghost=True)

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
        """Send Pac and ghosts back to spawn (after being caught)."""
        jx = int(self.rng.integers(-2, 3))
        self._pac.col, self._pac.row = self._nearest_passable(
            self._W // 2 + jx, self._H - 6)
        self._pac.prog = 0.0
        self._pac.dcol = self._pac.drow = 0
        anchors = [(2, 2), (self._W - 3, 2), (2, self._H - 3), (self._W - 3, self._H - 3)]
        self.rng.shuffle(anchors)
        for g, a in zip(self._ghosts, anchors):
            g.col, g.row = self._nearest_passable(*a)
            g.prog = 0.0
            g.dcol = g.drow = 0
            g.mode = 0.0
            g.respawn = 0.0
        self._fright = 0.0

    def _reset_board(self) -> None:
        """Refill pellets for a fresh level and shift the palette."""
        self._dots = self._dots0.copy()
        self._dot_dirty = True
        self._level_timer = 0.0
        self._hue = float((self._hue + self.rng.uniform(0.1, 0.35)) % 1.0)
        self._flash = max(self._flash, 0.7)

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

        # eaten ghost respawn
        for g in self._ghosts:
            if g.mode < 0.0:
                g.respawn -= dt
                if g.respawn <= 0.0:
                    g.col, g.row = self._nearest_passable(self._W // 2, self._H // 2 - 4)
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

        # level completion / safety refill
        self._level_timer += dt
        remaining = float((self._dots > 0.0).sum())
        if remaining < 0.05 * self._dot_total or self._level_timer > 90.0:
            self._reset_board()

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

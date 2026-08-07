"""Tetris — a real falling-block simulation rendered as a neon light-show.

Same architecture as the rest of the pack: an actual Tetris game runs on the
CPU (a real 10x20 well, the seven tetromino shapes, gravity, rotation, line
clears) and the shader renders the true board state. A self-playing AI
evaluates every rotation/column for each piece (minimising holes/height/
bumpiness, rewarding line clears) and executes the best placement.

Play is deliberately fast: the AI snaps the piece to its planned rotation and
column at a high handling rate, then **hard-drops** it. Each drop leaves a
streak trail down the columns it fell through and lands with a shockwave and
a screen-shake kick, so the board reads as a rapid-fire slam sequence rather
than a slow drift. Difficulty escalates with a level counter (shown as a lit
segment ladder beside the well) that drives gravity, and a topped-out board
detonates and restarts instead of silently jamming forever.

Presentation is holographic rather than flat: a receding perspective grid and
falling data-rain behind the well, corner brackets and a travelling scan line
on the frame, and blocks drawn as bevelled shells around a pulsing energy
core. Line clears fire a horizontal disintegration beam across each full row
before it collapses, and shatter into neon shrapnel.

Audio reactivity:
  - bass    -> gravity/handling surge, board pulse, level-ladder glow
  - mid     -> piece glow shimmer, palette drift, data-rain density
  - treble  -> shrapnel sparkle, scan-line and grid sparkle
  - beat    -> board flash, palette shift, brief speed surge

Randomised per activation: palette hue, starting bag order, board tempo
phase, and ``self.time`` — no two runs look or play alike from frame 0.
"""
from __future__ import annotations

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

_COLS = 10
_ROWS = 20
_NEX = 16
_NCLEAR = 4
_OFF = 1000.0

# Tetromino definitions: 4 rotation states, each a list of (col, row) cells
# in a 4x4 local grid (row 0 = top). Standard SRS-ish shapes (no true SRS
# kick tables — a simple kick search is good enough for a self-playing showcase).
_PIECES: dict[str, list[list[tuple[int, int]]]] = {
    'I': [[(0, 1), (1, 1), (2, 1), (3, 1)], [(2, 0), (2, 1), (2, 2), (2, 3)],
          [(0, 2), (1, 2), (2, 2), (3, 2)], [(1, 0), (1, 1), (1, 2), (1, 3)]],
    'O': [[(1, 0), (2, 0), (1, 1), (2, 1)]] * 4,
    'T': [[(1, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (1, 1), (2, 1), (1, 2)],
          [(0, 1), (1, 1), (2, 1), (1, 2)], [(1, 0), (0, 1), (1, 1), (1, 2)]],
    'S': [[(1, 0), (2, 0), (0, 1), (1, 1)], [(1, 0), (1, 1), (2, 1), (2, 2)],
          [(1, 1), (2, 1), (0, 2), (1, 2)], [(0, 0), (0, 1), (1, 1), (1, 2)]],
    'Z': [[(0, 0), (1, 0), (1, 1), (2, 1)], [(2, 0), (1, 1), (2, 1), (1, 2)],
          [(0, 1), (1, 1), (1, 2), (2, 2)], [(1, 0), (0, 1), (1, 1), (0, 2)]],
    'J': [[(0, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (2, 0), (1, 1), (1, 2)],
          [(0, 1), (1, 1), (2, 1), (2, 2)], [(1, 0), (1, 1), (0, 2), (1, 2)]],
    'L': [[(2, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (1, 1), (1, 2), (2, 2)],
          [(0, 1), (1, 1), (2, 1), (0, 2)], [(0, 0), (1, 0), (1, 1), (1, 2)]],
}
_PIECE_NAMES = tuple(_PIECES.keys())
_PIECE_COLORS: dict[str, tuple[float, float, float]] = {
    'I': (0.20, 0.95, 1.00), 'O': (1.00, 0.90, 0.20), 'T': (0.80, 0.25, 1.00),
    'S': (0.30, 1.00, 0.40), 'Z': (1.00, 0.25, 0.30), 'J': (0.25, 0.45, 1.00),
    'L': (1.00, 0.60, 0.15),
}

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
uniform float iBass;
uniform float iMid;
uniform float iTreble;
uniform float iBeat;
uniform float iFlash;
uniform float iHue;
uniform float iGlitter;
uniform float iOriginX;
uniform float iOriginTopY;
uniform float iCell;
uniform vec2  iGrid;          // (cols, rows)
uniform sampler2D iBoard;     // rgb = block colour, a>0.5 = occupied
uniform vec4  iActive[4];     // active piece cells: (col, row, _, _) in board coords
uniform vec3  iActiveCol;
uniform vec4  iGhost[4];      // ghost/landing preview cells
uniform vec4  iNext[4];       // next-piece preview cells (local 0..3 space)
uniform vec3  iNextCol;
uniform vec3  iShard[16];     // (x, y, age); age<0 inactive — line-clear shrapnel
uniform vec3  iShardCol[16];
uniform float iTrailCols[4];  // columns the last hard drop fell through; <0 unused
uniform vec2  iTrailY;        // (fromRow, toRow) of the last hard drop
uniform float iTrailAge;      // <0 inactive
uniform vec3  iTrailCol;
uniform float iSweepRows[4];  // rows being disintegrated; <0 unused
uniform float iSweepAge;
uniform float iSlam;          // landing-impact envelope [0,1]
uniform vec2  iSlamAt;        // (col, row) of the impact
uniform float iLevel;
in  vec2 v_uv;
out vec4 fragColor;

#define PI  3.14159265
#define TAU 6.28318530

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

// Holographic backdrop: receding grid below the horizon, data-rain above.
// Every time term is wrapped before it reaches fract()/floor() — iTime can
// start as high as 10000 by design, and raw values that large lose enough
// float32 fractional precision to collapse the grid into flat bands.
vec3 backdrop(vec2 uv, float A, float tb) {
    vec2 p = (uv - 0.5) * vec2(A, 1.0);
    vec3 col = mix(vec3(0.004, 0.007, 0.020), vec3(0.014, 0.006, 0.030), uv.y);

    // Thin grid lines only: the mask must peak where fract() is near 0.5 and
    // fall off immediately. Subtracting a constant inside smoothstep instead
    // (as an earlier version did) inverts the intent — it lights ~80% of the
    // cell and the "grid" floods the screen as a pale wash.
    float horizon = 0.24;
    if (p.y < horizon) {
        float depth = max(horizon - p.y, 0.012);
        float persp = 0.05 / depth;
        float lx = smoothstep(0.05, 0.0, abs(fract(p.x * persp) - 0.5));
        float lz = smoothstep(0.05, 0.0, abs(fract(persp + tb * 0.45) - 0.5));
        float fade = smoothstep(0.0, 0.14, depth) * smoothstep(0.80, 0.20, depth);
        col += mix(vec3(0.10, 0.45, 1.0), palette(iHue), 0.45)
             * (lx * 0.5 + lz) * fade * (0.11 + iBass * 0.10);
    }

    // Data rain: sparse falling glyph columns, denser with mid energy.
    float colId = floor(p.x * 30.0);
    float seed = hash21(vec2(colId, 7.0));
    if (seed > 0.90 - iMid * 0.06) {
        float speed = 0.4 + seed * 0.7;
        float yy = fract(p.y * 1.3 - tb * speed + seed * 10.0);
        float glyph = step(0.5, hash21(vec2(colId, floor(p.y * 40.0 - tb * speed * 26.0))));
        float lane = smoothstep(0.5, 0.25, abs(fract(p.x * 30.0) - 0.5));
        col += vec3(0.20, 0.85, 0.70) * glyph * lane
             * smoothstep(0.0, 0.40, yy) * smoothstep(1.0, 0.60, yy) * 0.11;
    }

    float sh = hash21(floor(uv * vec2(A, 1.0) * 90.0));
    if (sh > 0.978) {
        float tw = 0.5 + 0.5 * sin(tb * (2.0 + sh * 6.0) + sh * 40.0);
        col += vec3(0.6, 0.7, 0.9) * tw * (0.25 + iTreble * 0.5);
    }
    return col;
}

// One block cell in local space q ~ [-0.5,0.5]^2: bevelled shell + energy core.
vec3 drawBlock(vec2 q, vec3 c, float pulse, float phase) {
    vec2 aq = abs(q);
    float m = max(aq.x, aq.y);
    float shell = smoothstep(0.50, 0.465, m);
    float inner = smoothstep(0.375, 0.335, m);
    vec3 o = c * 2.0 * (shell - inner);                 // bright bevel rim
    o += c * 0.26 * inner;                              // dim inner face
    float core = smoothstep(0.26, 0.0, aq.x + aq.y);    // pulsing diamond core
    o += c * core * (0.45 + pulse);
    // Travelling scanline across the face.
    o += c * inner * 0.35
       * smoothstep(0.03, 0.0, abs(fract(q.y * 2.0 + phase) - 0.5) - 0.44);
    return o;
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float t = iTime;
    float tb = mod(t, 600.0);          // bounded time for all fract()/floor() use

    // Landing kick: a brief directional shake, strongest right at impact.
    vec2 shake = vec2(sin(tb * 61.0), sin(tb * 47.0 + 1.7))
               * iSlam * iSlam * iCell * 0.22;
    vec2 frag = v_uv * iResolution + shake;

    vec3 col = backdrop(v_uv, A, tb);

    vec2 gc = vec2((frag.x - iOriginX) / iCell, (iOriginTopY - frag.y) / iCell);
    ivec2 cell = ivec2(floor(gc));
    vec2 f = fract(gc) - 0.5;
    bool inBoard = (gc.x >= 0.0 && gc.y >= 0.0 && gc.x < iGrid.x && gc.y < iGrid.y);

    if (inBoard) {
        vec4 b = texelFetch(iBoard, cell, 0);
        float pulse = 0.5 + 0.5 * sin(tb * 2.0 + float(cell.y) * 0.3);
        if (b.a > 0.5) {
            col += drawBlock(f, b.rgb, pulse * 0.3 + iBass * 0.3, tb * 0.4);
        } else {
            col += vec3(0.05, 0.08, 0.14) * 0.15;
            col += vec3(0.10, 0.22, 0.40)
                 * smoothstep(0.49, 0.5, max(abs(f.x), abs(f.y))) * 0.30;
        }
    }

    // Hard-drop streaks: the columns the piece just fell through.
    if (iTrailAge >= 0.0) {
        float fade = pow(1.0 - iTrailAge, 1.7);
        for (int i = 0; i < 4; i++) {
            if (iTrailCols[i] < 0.0) continue;
            float dx = abs(gc.x - (iTrailCols[i] + 0.5));
            if (dx > 0.5) continue;
            float inSpan = step(iTrailY.x, gc.y) * step(gc.y, iTrailY.y + 1.0);
            float taper = smoothstep(iTrailY.x - 1.0, iTrailY.y, gc.y);
            col += iTrailCol * smoothstep(0.5, 0.0, dx) * inSpan * taper * fade * 1.1;
        }
    }

    // Line-clear disintegration beam sweeping across each full row.
    if (iSweepAge >= 0.0) {
        for (int i = 0; i < 4; i++) {
            if (iSweepRows[i] < 0.0) continue;
            float dy = abs(gc.y - (iSweepRows[i] + 0.5));
            if (dy > 0.6) continue;
            float head = iSweepAge * (iGrid.x + 2.0) - 1.0;
            float along = head - gc.x;
            float beam = smoothstep(0.0, 1.2, along) * smoothstep(4.5, 0.6, along);
            float edge = smoothstep(0.6, 0.0, abs(along)) * 2.2;
            col += mix(vec3(1.0), palette(iHue + 0.3), 0.35)
                 * smoothstep(0.6, 0.0, dy) * (beam * 1.4 + edge)
                 * (1.0 - iSweepAge * 0.4);
        }
    }

    // Well frame: neon border, corner brackets, travelling scan line.
    float bx = min(gc.x, iGrid.x - gc.x) * iCell;
    float by = min(gc.y, iGrid.y - gc.y) * iCell;
    float borderDist = inBoard ? min(bx, by) : 1e6;
    vec3 borderCol = mix(vec3(0.2, 0.6, 1.0), palette(iHue), 0.5);
    col += borderCol * smoothstep(3.0, 0.0, abs(borderDist)) * (0.8 + iBass * 0.6);
    if (inBoard) {
        float cx = min(gc.x, iGrid.x - gc.x);
        float cy = min(gc.y, iGrid.y - gc.y);
        float bracket = step(cx, 3.0) * step(cy, 0.35) + step(cy, 3.0) * step(cx, 0.35);
        col += borderCol * clamp(bracket, 0.0, 1.0) * (1.2 + iBeat * 0.8);
        float scan = abs(gc.y - fract(tb * 0.12) * iGrid.y);
        col += borderCol * smoothstep(0.6, 0.0, scan) * 0.16 * (0.5 + iTreble * 0.8);
    }

    // Ghost (landing-preview) cells: dashed outline.
    for (int i = 0; i < 4; i++) {
        vec2 gp = iGhost[i].xy;
        if (gp.x < -0.5) continue;
        vec2 lf = gc - (gp + 0.5);
        float ring = smoothstep(0.5, 0.46, max(abs(lf.x), abs(lf.y)))
                   * (1.0 - smoothstep(0.42, 0.38, max(abs(lf.x), abs(lf.y))));
        float dash = step(0.35, fract((lf.x + lf.y) * 4.0 + tb * 0.5));
        col += iActiveCol * ring * dash * 0.55;
    }

    // Active falling piece.
    for (int i = 0; i < 4; i++) {
        vec2 ap = iActive[i].xy;
        if (ap.x < -0.5) continue;
        vec2 lf = gc - (ap + 0.5);
        if (abs(lf.x) < 0.5 && abs(lf.y) < 0.5) {
            col += drawBlock(lf, iActiveCol, 0.45 + iBass * 0.5 + iBeat * 0.5, tb * 0.9);
            col += iActiveCol * 0.25 * (0.5 + iMid * 0.6);
        }
    }

    // Impact shockwave from the last landing.
    if (iSlam > 0.0) {
        vec2 d = gc - (iSlamAt + vec2(0.5, 0.5));
        float rad = (1.0 - iSlam) * 9.0;
        float ring = smoothstep(1.4, 0.0, abs(length(d) - rad));
        col += mix(vec3(1.0, 0.95, 0.8), palette(iHue + 0.2), 0.4) * ring * iSlam * 0.85;
    }

    // Next-piece preview in a holographic panel right of the well.
    float previewX = iGrid.x + 2.5;
    for (int i = 0; i < 4; i++) {
        vec2 np = iNext[i].xy;
        if (np.x < -0.5) continue;
        vec2 lf = gc - vec2(previewX + np.x, 2.0 + np.y);
        if (abs(lf.x) < 0.5 && abs(lf.y) < 0.5) {
            col += drawBlock(lf, iNextCol, 0.3 + iMid * 0.4, tb * 0.6);
        }
    }
    float panelX = abs(gc.x - (previewX + 1.5)) - 2.6;
    float panelY = abs(gc.y - 3.0) - 2.6;
    col += iNextCol * smoothstep(0.12, 0.0, abs(max(panelX, panelY))) * 0.45;
    col += iNextCol * smoothstep(0.0, -2.0, max(panelX, panelY)) * 0.020;

    // Level ladder left of the well: lit segments count the current level.
    float lvlX = -2.3;
    if (abs(gc.x - lvlX) < 0.6 && gc.y >= 0.0 && gc.y <= iGrid.y) {
        float up = iGrid.y - gc.y;
        float seg = floor(up / 1.5);
        if (seg >= 0.0 && seg < 12.0) {
            float fy = abs(fract(up / 1.5) - 0.5);
            float m = smoothstep(0.44, 0.28, fy) * smoothstep(0.55, 0.40, abs(gc.x - lvlX));
            vec3 sc = mix(vec3(0.15, 0.9, 1.0), vec3(1.0, 0.3, 0.45), seg / 11.0);
            float on = step(seg, iLevel - 1.0);
            col += sc * m * mix(0.05, 0.95 + iBass * 0.5, on);
        }
    }

    // Line-clear shrapnel.
    for (int i = 0; i < 16; i++) {
        if (iShard[i].z < 0.0) continue;
        vec2 d = gc - iShard[i].xy;
        float age = iShard[i].z;
        float m = smoothstep(0.35, 0.0, length(d)) * (1.0 - age);
        col += iShardCol[i] * m * (1.2 + iTreble * 0.8);
    }

    col += palette(iHue + 0.5) * iFlash * 0.25;
    float sp = hash21(floor(frag * 0.5) + floor(tb * 20.0));
    col += vec3(0.85, 0.92, 1.0) * step(0.994, sp) * (0.3 + iTreble * 0.8) * iGlitter;

    float vig = smoothstep(1.5, 0.2, length(v_uv - 0.5));
    col *= vig;
    col = col / (1.0 + col * 0.5);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.85));
    fragColor = vec4(col, 1.0);
}
"""


class _Piece:
    __slots__ = ('kind', 'rot', 'col', 'row')

    def __init__(self, kind: str, col: int = 3, row: int = -2) -> None:
        self.kind = kind
        self.rot = 0
        self.col = col
        self.row = row

    def cells(self, rot: int | None = None) -> list[tuple[int, int]]:
        r = self.rot if rot is None else rot % 4
        return [(self.col + c, self.row + rr) for c, rr in _PIECES[self.kind][r]]


class Tetris(BaseEffect):
    """Self-playing neon Tetris: a real falling-block simulation on the CPU."""

    NAME = 'Tetris'
    AUTHOR = 'unicorn-viz'
    TAGS = ['games', 'arcade', 'neon', 'intense', 'energetic', 'hard']
    PING_PONG_FRIENDS = [
        'Breakout', 'Neon Pac', 'Galaga', 'Joust',
        'Tron Grid', 'Laser Tunnel', 'Audio Spectrum', 'Audio Waveforms',
    ]

    _FALL_BASE = 7.0          # cells/sec baseline gravity (level 1)
    _HANDLING = 26.0          # rotation/slide steps per second
    _THINK = 0.10             # pause once aligned, before the hard drop
    _LOCK_PAUSE = 0.13        # beat after a lock, so the slam is legible
    _LOCK_DELAY = 0.30        # fallback lock when the plan can't be reached

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.3)),
            'glitter': float(self.config.get('glitter', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._board_tex = self.ctx.texture((_COLS, _ROWS), 4, dtype='f4')
        self._board_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._board = np.zeros((_ROWS, _COLS, 4), dtype=np.float32)
        self._board_dirty = True

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._flash = 0.0
        self._hue = float(self.rng.uniform(0.0, 1.0))

        self._bag: list[str] = []
        self._refill_bag()
        self._current = _Piece(self._draw_bag())
        self._next_kind = self._draw_bag()
        self._fall_t = 0.0
        self._lock_t = 0.0
        self._align_t = 0.0
        self._think_t = 0.0
        self._lock_pause = 0.0
        self._shards = np.full((_NEX, 3), -1.0, dtype=np.float32)
        self._shard_vel = np.zeros((_NEX, 3), dtype=np.float32)
        self._shard_col = np.zeros((_NEX, 3), dtype=np.float32)
        self._clear_flash = 0.0

        self._lines = 0
        self._level = 1
        self._slam = 0.0
        self._slam_at = (4.0, 19.0)
        self._trail_cols: list[float] = []
        self._trail_y = (0.0, 0.0)
        self._trail_age = -1.0
        self._trail_col = (1.0, 1.0, 1.0)
        self._sweep_rows: list[float] = []
        self._sweep_age = -1.0

        self._target = self._plan_move(self._current)

    # ------------------------------------------------------------------ bag
    def _refill_bag(self) -> None:
        order = list(_PIECE_NAMES)
        self.rng.shuffle(order)
        self._bag.extend(order)

    def _draw_bag(self) -> str:
        if not self._bag:
            self._refill_bag()
        return self._bag.pop(0)

    # --------------------------------------------------------------- board
    def _collides(self, cells: list[tuple[int, int]]) -> bool:
        for c, r in cells:
            if c < 0 or c >= _COLS or r >= _ROWS:
                return True
            if r >= 0 and self._board[r, c, 3] > 0.5:
                return True
        return False

    def _height_map(self, board: np.ndarray) -> np.ndarray:
        occupied = board[:, :, 3] > 0.5
        heights = np.zeros(_COLS, dtype=np.int32)
        for c in range(_COLS):
            col = np.where(occupied[:, c])[0]
            heights[c] = _ROWS - int(col[0]) if col.size else 0
        return heights

    def _score_placement(self, board: np.ndarray) -> float:
        occ = board[:, :, 3] > 0.5
        heights = self._height_map(board)
        agg_height = float(heights.sum())
        bumpiness = float(np.abs(np.diff(heights)).sum())
        holes = 0
        for c in range(_COLS):
            col = occ[:, c]
            filled_idx = np.where(col)[0]
            if filled_idx.size:
                holes += int((~col[filled_idx[0]:]).sum())
        full_rows = int(occ.all(axis=1).sum())
        return (-0.51 * agg_height - 0.36 * holes - 0.18 * bumpiness
                + 0.76 * (full_rows ** 1.5))

    def _plan_move(self, piece: _Piece) -> dict:
        """Evaluate every rotation/column and pick the best-scoring landing."""
        best = None
        best_score = -1e18
        for rot in range(4):
            shape = _PIECES[piece.kind][rot]
            cols_used = sorted({c for c, _ in shape})
            min_c, max_c = cols_used[0], cols_used[-1]
            for col in range(-min_c, _COLS - max_c):
                test = _Piece(piece.kind, col=col, row=-2)
                test.rot = rot
                if self._collides(test.cells()):
                    continue
                while not self._collides([(c, r + 1) for c, r in test.cells()]):
                    test.row += 1
                sim = self._board.copy()
                for c, r in test.cells():
                    if 0 <= r < _ROWS:
                        sim[r, c, :3] = _PIECE_COLORS[piece.kind]
                        sim[r, c, 3] = 1.0
                score = self._score_placement(sim)
                if score > best_score:
                    best_score = score
                    best = {'rot': rot, 'col': col}
        return best or {'rot': 0, 'col': piece.col}

    def _hard_drop(self) -> None:
        """Slam the piece to its landing row, leaving a streak and a shockwave."""
        piece = self._current
        start_row = max(piece.row, 0)
        while not self._collides([(c, r + 1) for c, r in piece.cells()]):
            piece.row += 1
        cells = piece.cells()
        self._trail_cols = sorted({float(c) for c, _ in cells})[:4]
        self._trail_y = (float(start_row), float(max(r for _, r in cells)))
        self._trail_age = 0.0
        self._trail_col = _PIECE_COLORS[piece.kind]
        self._slam = 1.0
        self._slam_at = (float(np.mean([c for c, _ in cells])),
                         float(max(r for _, r in cells)))
        self._lock_piece()

    def _reset_board(self) -> None:
        """Top-out: detonate the stack and start a fresh well."""
        for r in range(_ROWS - 1, _ROWS - 3, -1):
            self._spawn_shards(r)
        self._board[:] = 0.0
        self._board_dirty = True
        self._flash = max(self._flash, 1.0)
        self._clear_flash = 1.0
        self._lines = 0
        self._level = 1
        self._hue = float((self._hue + self.rng.uniform(0.2, 0.4)) % 1.0)

    def _lock_piece(self) -> None:
        cells = self._current.cells()
        color = _PIECE_COLORS[self._current.kind]
        for c, r in cells:
            if 0 <= r < _ROWS and 0 <= c < _COLS:
                self._board[r, c, :3] = color
                self._board[r, c, 3] = 1.0
        self._board_dirty = True
        self._clear_lines()
        self._current = _Piece(self._next_kind)
        self._next_kind = self._draw_bag()
        if self._collides(self._current.cells()):
            self._reset_board()
            self._current = _Piece(self._next_kind)
            self._next_kind = self._draw_bag()
        self._target = self._plan_move(self._current)
        self._fall_t = 0.0
        self._lock_t = 0.0
        self._think_t = 0.0
        self._align_t = 0.0
        self._lock_pause = self._LOCK_PAUSE

    def _clear_lines(self) -> None:
        occ = self._board[:, :, 3] > 0.5
        full = np.where(occ.all(axis=1))[0]
        if full.size == 0:
            return
        for r in full:
            self._spawn_shards(int(r))
        self._sweep_rows = [float(r) for r in full[:_NCLEAR]]
        self._sweep_age = 0.0
        keep = np.ones(_ROWS, dtype=bool)
        keep[full] = False
        remaining = self._board[keep]
        pad = np.zeros((int(full.size), _COLS, 4), dtype=np.float32)
        self._board = np.concatenate([pad, remaining], axis=0)
        self._board_dirty = True
        self._flash = max(self._flash, 0.6 + 0.15 * float(full.size))
        self._clear_flash = 1.0
        self._hue = float((self._hue + self.rng.uniform(0.08, 0.22)) % 1.0)
        self._lines += int(full.size)
        self._level = min(12, 1 + self._lines // 8)

    def _spawn_shards(self, row: int) -> None:
        free = np.where(self._shards[:, 2] < 0.0)[0]
        if free.size == 0:
            return
        cols = list(range(_COLS))
        self.rng.shuffle(cols)
        for i, c in enumerate(cols[:len(free)]):
            slot = free[i]
            self._shards[slot] = (c + 0.5, row + 0.5, 0.0)
            ang = float(self.rng.uniform(0, 2 * np.pi))
            spd = float(self.rng.uniform(2.0, 5.0))
            self._shard_vel[slot] = (np.cos(ang) * spd, -abs(np.sin(ang) * spd) - 2.0, 0.0)
            self._shard_col[slot] = (self._board[row, c, :3]
                                     if self._board[row, c, 3] > 0 else (1.0, 1.0, 1.0))

    # ---------------------------------------------------------------- update
    def update(self, dt: float, audio: AudioData) -> None:
        """Advance audio envelopes and the whole Tetris simulation."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 2.0))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 2.0))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 2.0))
        if audio.beat > 0.5:
            self._beat = 1.0
            self._flash = max(self._flash, 0.3)
        self._beat = max(0.0, self._beat - dt * 2.5)
        self._flash = max(0.0, self._flash - dt * 1.8)
        self._clear_flash = max(0.0, self._clear_flash - dt * 2.0)
        self._slam = max(0.0, self._slam - dt * 3.4)
        if self._trail_age >= 0.0:
            self._trail_age += dt * 3.2
            if self._trail_age > 1.0:
                self._trail_age = -1.0
        if self._sweep_age >= 0.0:
            self._sweep_age += dt * 2.6
            if self._sweep_age > 1.0:
                self._sweep_age = -1.0
                self._sweep_rows = []

        tempo = max(0.05, float(self.parameters['speed'])) * (1.0 + self._bass * 0.6)
        level_gain = 1.0 + (self._level - 1) * 0.16

        self._lock_pause = max(0.0, self._lock_pause - dt)
        if self._lock_pause <= 0.0:
            self._step_piece(dt, tempo * level_gain)

        # Shrapnel physics (board space: +y is downward, so gravity is +y).
        act = self._shards[:, 2] >= 0.0
        self._shard_vel[act, 1] += dt * 9.0
        self._shards[act, 0] += self._shard_vel[act, 0] * dt
        self._shards[act, 1] += self._shard_vel[act, 1] * dt
        self._shards[act, 2] += dt * 1.1
        self._shards[self._shards[:, 2] > 1.0] = -1.0

    def _step_piece(self, dt: float, tempo: float) -> None:
        """Snap the piece toward its plan, then hard-drop once it is aligned."""
        want_rot = self._target['rot']
        want_col = self._target['col']

        self._align_t += dt * self._HANDLING * tempo
        steps = int(self._align_t)
        self._align_t -= steps
        for _ in range(steps):
            if self._current.rot != want_rot:
                test = _Piece(self._current.kind, col=self._current.col,
                              row=self._current.row)
                test.rot = (self._current.rot + 1) % 4
                if self._collides(test.cells()):
                    break
                self._current.rot = test.rot
            elif self._current.col != want_col:
                step = 1 if want_col > self._current.col else -1
                test = _Piece(self._current.kind, col=self._current.col + step,
                              row=self._current.row)
                test.rot = self._current.rot
                if self._collides(test.cells()):
                    break
                self._current.col += step
            else:
                break

        self._fall_t += dt * self._FALL_BASE * tempo
        while self._fall_t >= 1.0:
            self._fall_t -= 1.0
            if self._collides([(c, r + 1) for c, r in self._current.cells()]):
                break
            self._current.row += 1

        aligned = (self._current.rot == want_rot and self._current.col == want_col)
        if aligned:
            self._think_t += dt
            if self._think_t >= self._THINK:
                self._hard_drop()
            return

        # Plan unreachable (blocked mid-slide): fall back to a normal lock so
        # the piece can never hang in the air forever.
        if self._collides([(c, r + 1) for c, r in self._current.cells()]):
            self._lock_t += dt
            if self._lock_t >= self._LOCK_DELAY:
                self._lock_piece()
        else:
            self._lock_t = 0.0

    # ---------------------------------------------------------------- render
    def render(self) -> None:
        """Upload board state and draw the neon well."""
        if self._board_dirty:
            self._board_tex.write(self._board.tobytes())
            self._board_dirty = False

        w, h = float(self.width), float(self.height)
        cell = min(w / (_COLS + 9.0), h / (_ROWS + 1.5))
        origin_x = (w - (_COLS + 3.0) * cell) * 0.5
        origin_top_y = (h + _ROWS * cell) * 0.5

        self._board_tex.use(location=0)
        p = self._prog
        p['iTime'].value = float(self.time)
        p['iResolution'].value = (w, h)
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iFlash'].value = self._flash + self._clear_flash * 0.5
        p['iHue'].value = self._hue
        p['iGlitter'].value = max(0.0, float(self.parameters['glitter']))
        p['iOriginX'].value = float(origin_x)
        p['iOriginTopY'].value = float(origin_top_y)
        p['iCell'].value = float(cell)
        p['iGrid'].value = (float(_COLS), float(_ROWS))
        p['iBoard'].value = 0
        p['iLevel'].value = float(self._level)
        p['iSlam'].value = float(self._slam)
        p['iSlamAt'].value = self._slam_at

        active_cells = self._current.cells()
        active = [(float(c), float(r), 0.0, 0.0) if r >= -2 else (_OFF, _OFF, 0.0, 0.0)
                  for c, r in active_cells]
        while len(active) < 4:
            active.append((_OFF, _OFF, 0.0, 0.0))
        p['iActive'].value = active
        p['iActiveCol'].value = _PIECE_COLORS[self._current.kind]

        ghost = _Piece(self._current.kind, col=self._current.col, row=self._current.row)
        ghost.rot = self._current.rot
        while not self._collides([(c, r + 1) for c, r in ghost.cells()]):
            ghost.row += 1
        ghost_val = [(float(c), float(r), 0.0, 0.0) for c, r in ghost.cells()]
        while len(ghost_val) < 4:
            ghost_val.append((_OFF, _OFF, 0.0, 0.0))
        p['iGhost'].value = ghost_val

        next_shape = _PIECES[self._next_kind][0]
        next_val = [(float(c), float(r), 0.0, 0.0) for c, r in next_shape]
        while len(next_val) < 4:
            next_val.append((_OFF, _OFF, 0.0, 0.0))
        p['iNext'].value = next_val
        p['iNextCol'].value = _PIECE_COLORS[self._next_kind]

        p['iShard'].value = [tuple(map(float, s)) for s in self._shards]
        p['iShardCol'].value = [tuple(map(float, s)) for s in self._shard_col]

        trail = list(self._trail_cols[:4]) + [-1.0] * (4 - len(self._trail_cols[:4]))
        p['iTrailCols'].value = [float(c) for c in trail]
        p['iTrailY'].value = self._trail_y
        p['iTrailAge'].value = float(self._trail_age)
        p['iTrailCol'].value = self._trail_col

        sweep = list(self._sweep_rows[:_NCLEAR]) + \
            [-1.0] * (_NCLEAR - len(self._sweep_rows[:_NCLEAR]))
        p['iSweepRows'].value = [float(r) for r in sweep]
        p['iSweepAge'].value = float(self._sweep_age)

        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
        self._board_tex.release()

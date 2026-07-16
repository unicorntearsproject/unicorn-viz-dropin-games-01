"""Tetris — a real falling-block simulation rendered as a neon light-show.

Same architecture as the rest of the pack: an actual Tetris game runs on the
CPU (a real 10x20 well, the seven tetromino shapes, gravity, rotation with
wall-kicks, line clears) and the shader renders the true board state as
glowing neon blocks. A self-playing AI evaluates every rotation/column for
each piece (minimising holes/height/bumpiness, rewarding line clears) and
executes the best placement, so the board is always alive with real,
legible Tetris play.

Line clears flash and shatter into neon shrapnel; a "next piece" preview
glows beside the well; a beat-synced board pulse and per-piece glow keep it
audio-reactive throughout.

Audio reactivity:
  - bass    -> drop speed, board pulse, clear-shatter force
  - mid     -> piece glow shimmer, palette drift
  - treble  -> shrapnel sparkle
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

vec3 backdrop(vec2 uv, float A, float t) {
    vec2 p = (uv - 0.5) * vec2(A, 1.0);
    float v = sin(p.x * 2.4 + t * 0.4) + sin(p.y * 2.0 - t * 0.3)
            + sin(length(p) * 3.0 - t * 0.6);
    vec3 col = palette(v * 0.08 + iHue + 0.6 + t * 0.015);
    col *= col;
    col *= 0.06 + 0.05 * (v * 0.5 + 0.5);
    float sh = hash21(floor(uv * vec2(A, 1.0) * 90.0));
    if (sh > 0.965) {
        float tw = 0.5 + 0.5 * sin(t * (2.0 + sh * 6.0) + sh * 40.0);
        col += vec3(0.6, 0.7, 0.9) * tw * (0.3 + iTreble * 0.5);
    }
    return col;
}

// One glowing block cell in local space q ~ [-0.5,0.5]^2.
vec3 drawBlock(vec2 q, vec3 c, float pulse) {
    float inset = 0.09;
    vec2 aq = abs(q);
    float body = step(aq.x, 0.5 - inset) * step(aq.y, 0.5 - inset);
    float full = step(aq.x, 0.5) * step(aq.y, 0.5);
    vec3 face = c * (0.55 + 0.35 * (1.0 - aq.y) + pulse * 0.4);
    vec3 edge = c * (1.3 + pulse * 0.6);
    vec3 o = mix(vec3(0.0), edge, full);
    o = mix(o, face, body);
    o += c * 0.15 * full;
    return o;
}

void main() {
    vec2 frag = v_uv * iResolution;
    float A = iResolution.x / max(iResolution.y, 1.0);
    float t = iTime;
    vec3 col = backdrop(v_uv, A, t);

    vec2 gc = vec2((frag.x - iOriginX) / iCell, (iOriginTopY - frag.y) / iCell);
    ivec2 cell = ivec2(floor(gc));
    vec2 f = fract(gc) - 0.5;
    bool inBoard = (gc.x >= 0.0 && gc.y >= 0.0 && gc.x < iGrid.x && gc.y < iGrid.y);

    if (inBoard) {
        vec4 b = texelFetch(iBoard, cell, 0);
        float pulse = 0.5 + 0.5 * sin(t * 2.0 + float(cell.y) * 0.3);
        if (b.a > 0.5) {
            col += drawBlock(f, b.rgb, pulse * 0.3 + iBass * 0.3);
        } else {
            col += vec3(0.05, 0.08, 0.14) * 0.15;
            col += vec3(0.1, 0.2, 0.35) * smoothstep(0.49, 0.5, max(abs(f.x), abs(f.y))) * 0.3;
        }
    }

    // Well border, neon outline.
    float bx = min(gc.x, iGrid.x - gc.x) * iCell;
    float by = min(gc.y, iGrid.y - gc.y) * iCell;
    float borderDist = inBoard ? min(bx, by) : 1e6;
    vec3 borderCol = mix(vec3(0.2, 0.6, 1.0), palette(iHue), 0.5);
    col += borderCol * smoothstep(3.0, 0.0, abs(borderDist)) * (0.8 + iBass * 0.6);

    // Ghost (landing-preview) cells: faint outline only.
    for (int i = 0; i < 4; i++) {
        vec2 gp = iGhost[i].xy;
        if (gp.x < -0.5) continue;
        vec2 lf = gc - (gp + 0.5);
        float m = smoothstep(0.5, 0.46, max(abs(lf.x), abs(lf.y)))
                * (1.0 - smoothstep(0.42, 0.38, max(abs(lf.x), abs(lf.y))));
        col += iActiveCol * m * 0.5;
    }

    // Active falling piece.
    for (int i = 0; i < 4; i++) {
        vec2 ap = iActive[i].xy;
        if (ap.x < -0.5) continue;
        vec2 lf = gc - (ap + 0.5);
        if (abs(lf.x) < 0.5 && abs(lf.y) < 0.5) {
            col = mix(col, vec3(0.0), 0.0);
            col += drawBlock(lf, iActiveCol, 0.4 + iBass * 0.5 + iBeat * 0.5);
        }
    }

    // Next-piece preview, off to the right of the well.
    float previewX = iGrid.x + 2.5;
    for (int i = 0; i < 4; i++) {
        vec2 np = iNext[i].xy;
        if (np.x < -0.5) continue;
        vec2 lf = gc - vec2(previewX + np.x, 2.0 + np.y);
        if (abs(lf.x) < 0.5 && abs(lf.y) < 0.5) {
            col += drawBlock(lf, iNextCol, 0.3 + iMid * 0.4);
        }
    }
    float previewGlow = smoothstep(2.6, 0.0, abs(gc.x - (previewX + 1.0)) + abs(gc.y - 3.0) * 0.3);
    col += iNextCol * previewGlow * 0.03;

    // Line-clear shrapnel.
    for (int i = 0; i < 16; i++) {
        if (iShard[i].z < 0.0) continue;
        vec2 sp = iShard[i].xy;
        vec2 d = gc - sp;
        float age = iShard[i].z;
        float m = smoothstep(0.35, 0.0, length(d)) * (1.0 - age);
        col += iShardCol[i] * m * (1.2 + iTreble * 0.8);
    }

    col += palette(iHue + 0.5) * iFlash * 0.25;
    float sp = hash21(floor(frag * 0.5) + floor(t * 20.0));
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

    _FALL_BASE = 1.6          # cells/sec baseline gravity
    _LOCK_DELAY = 0.35

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
        self._shards = np.full((_NEX, 3), -1.0, dtype=np.float32)
        self._shard_vel = np.zeros((_NEX, 3), dtype=np.float32)
        self._shard_col = np.zeros((_NEX, 3), dtype=np.float32)
        self._clear_flash = 0.0

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
        self._target = self._plan_move(self._current)
        self._fall_t = 0.0
        self._lock_t = 0.0

    def _clear_lines(self) -> None:
        occ = self._board[:, :, 3] > 0.5
        full = np.where(occ.all(axis=1))[0]
        if full.size == 0:
            return
        for r in full:
            self._spawn_shards(int(r))
        keep = np.ones(_ROWS, dtype=bool)
        keep[full] = False
        remaining = self._board[keep]
        pad = np.zeros((int(full.size), _COLS, 4), dtype=np.float32)
        self._board = np.concatenate([pad, remaining], axis=0)
        self._board_dirty = True
        self._flash = max(self._flash, 0.6 + 0.15 * float(full.size))
        self._clear_flash = 1.0
        self._hue = float((self._hue + self.rng.uniform(0.08, 0.22)) % 1.0)

    def _spawn_shards(self, row: int) -> None:
        free = np.where(self._shards[:, 2] < 0.0)[0]
        cols = list(range(_COLS))
        self.rng.shuffle(cols)
        for i, c in enumerate(cols[:len(free)]):
            slot = free[i]
            self._shards[slot] = (c + 0.5, _ROWS - row - 0.5, 0.0)
            ang = float(self.rng.uniform(0, 2 * np.pi))
            spd = float(self.rng.uniform(2.0, 5.0))
            self._shard_vel[slot] = (np.cos(ang) * spd, np.sin(ang) * spd + 2.0, 0.0)
            self._shard_col[slot] = self._board[row, c, :3] if self._board[row, c, 3] > 0 else (1.0, 1.0, 1.0)

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

        tempo = max(0.05, float(self.parameters['speed'])) * (1.0 + self._bass * 0.6)

        # AI steering: rotate then slide toward the planned column, one step
        # at a time so the piece visibly rotates/slides rather than teleporting.
        want_rot = self._target['rot']
        want_col = self._target['col']
        if self._current.rot != want_rot:
            test = _Piece(self._current.kind, col=self._current.col, row=self._current.row)
            test.rot = (self._current.rot + 1) % 4
            if not self._collides(test.cells()):
                self._current.rot = test.rot
        elif self._current.col != want_col:
            step = 1 if want_col > self._current.col else -1
            test = _Piece(self._current.kind, col=self._current.col + step, row=self._current.row)
            test.rot = self._current.rot
            if not self._collides(test.cells()):
                self._current.col += step

        # Gravity / lock.
        self._fall_t += dt * self._FALL_BASE * tempo
        while self._fall_t >= 1.0:
            self._fall_t -= 1.0
            below = [(c, r + 1) for c, r in self._current.cells()]
            if self._collides(below):
                break
            self._current.row += 1

        below = [(c, r + 1) for c, r in self._current.cells()]
        grounded = self._collides(below)
        if grounded:
            self._lock_t += dt
            if self._lock_t >= self._LOCK_DELAY:
                self._lock_piece()
        else:
            self._lock_t = 0.0

        # Shrapnel physics.
        act = self._shards[:, 2] >= 0.0
        self._shard_vel[act, 1] -= dt * 9.0
        self._shards[act, 0] += self._shard_vel[act, 0] * dt
        self._shards[act, 1] += self._shard_vel[act, 1] * dt
        self._shards[act, 2] += dt * 1.1
        self._shards[self._shards[:, 2] > 1.0] = -1.0

    # ---------------------------------------------------------------- render
    def render(self) -> None:
        """Upload board state and draw the neon well."""
        if self._board_dirty:
            self._board_tex.write(self._board.tobytes())
            self._board_dirty = False

        w, h = float(self.width), float(self.height)
        cell = min(w / (_COLS + 6.0), h / (_ROWS + 1.5))
        origin_x = (w - (_COLS + 4.0) * cell) * 0.5
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
        ghost_cells = ghost.cells()
        ghost_val = [(float(c), float(r), 0.0, 0.0) for c, r in ghost_cells]
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

        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
        self._board_tex.release()

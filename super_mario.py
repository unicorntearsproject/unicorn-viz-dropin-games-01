"""Super Mario — a real auto-running platformer, simulated then drawn.

Like the rest of this pack (see ``breakout.py`` for the pattern this follows),
nothing here is an animation loop. There is a level, a runner with mass, and
the runner plays it: the level is generated ahead of the camera and recycled
behind it, the runner carries velocity and gravity, and every jump is a
decision taken from what is actually in front of him -- a pit he would fall
into, a goomba he would walk into, a block he could open. He clears the gaps
because he reads them, not because a curve says jump here.

The level is a **tile heightmap**, generated one column at a time on the CPU
and handed to the shader as an array. That keeps the terrain honest -- a pit is
a column with no ground, and the runner's collision, the camera and the drawn
world all read the same numbers -- and it keeps the fragment shader to looking
up a column and shading it.

What the simulation actually runs:
  - Gravity, a jump impulse, and a variable-height jump (held longer on a
    stronger beat), with coyote-time so a jump taken a frame late still fires.
  - Goombas walking left; landing on one squashes it, walking into one costs
    the run a stumble.
  - Question blocks that open when hit from below, popping a coin that arcs.
  - Pipes he has to clear, and staircases he runs up.

Audio reactivity:
  - bass    -> run speed, jump strength, ground glow
  - mid     -> parallax drift, palette
  - treble  -> coin sparkle, brick grain
  - beat    -> a jump when one is available, block pop, flash

Randomised per activation: palette hue, level seed (so the terrain, pipes and
enemies differ every run), starting distance, and time-of-day sky.

Config keys (under ``[effects.SuperMario]``):
    speed, reactivity, glitter
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.15.0'

_COLS = 19              # ground columns handed to the shader (view + margin)
_VIEW_W = 15.0          # tiles visible across the screen
_NBLOCK = 10
_NENEMY = 8
_NCOIN = 10
_GROUND_Y = 2.0         # default ground top, in tiles from the bottom
_PIT = -5.0             # a column with no ground at all
_GRAVITY = 62.0         # tiles / s^2
_OFF = -99.0
_HEAD = 1.05            # tiles from the runner's feet to the top of his cap

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
uniform vec2  iResolution;
uniform float iTime;
uniform float iBass;
uniform float iMid;
uniform float iTreble;
uniform float iBeat;
uniform float iHue;
uniform float iGlitter;
uniform float iSky;            // per-activation time of day
uniform float iCamFrac;        // camera x within its tile, 0..1
uniform float iCamY;           // camera lift, tiles -- follows a high jump
uniform float iViewH;          // tiles visible vertically
uniform float iGround[19];     // ground top per column, tiles (_PIT = no ground)
uniform vec4  iBlock[10];      // x, y, kind (0 brick 1 question 2 spent), hit
uniform vec4  iEnemy[8];       // x, y, squash (0 alive .. 1 flat), face
uniform vec4  iCoin[10];       // x, y, life, spin
uniform vec4  iPipe[4];        // x, height, _, _
uniform vec3  iHero;           // x, y (feet), face
uniform float iHeroAir;        // 0 grounded, 1 airborne
uniform float iHeroStumble;
in  vec2 v_uv;
out vec4 fragColor;

#define TAU 6.28318530

float hash21(vec2 p) {
    p = mod(p, 512.0);
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }
float box(vec2 p, vec2 b) { vec2 d = abs(p) - b; return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0); }

// Ground top at a world column, with the camera's sub-tile offset applied.
float groundAt(float wx) {
    int i = int(clamp(floor(wx), 0.0, 18.0));
    return iGround[i];
}

// Sky, hills and clouds. Everything here parallaxes slower than the level,
// which is what gives the run its depth.
vec3 backdrop(vec2 w, float t) {
    float up = clamp(w.y / iViewH, 0.0, 1.0);
    vec3 lo, hi;
    if (iSky < 0.34)      { lo = vec3(0.55, 0.80, 1.00); hi = vec3(0.10, 0.30, 0.80); }
    else if (iSky < 0.67) { lo = vec3(1.00, 0.68, 0.38); hi = vec3(0.18, 0.16, 0.58); }
    else                  { lo = vec3(0.24, 0.16, 0.52); hi = vec3(0.04, 0.05, 0.24); }
    vec3 col = mix(lo, hi, pow(up, 0.8));
    col = mix(col, palette(iHue + 0.55), 0.12 + iMid * 0.10);

    // Rolling hills, two layers at different parallax rates.
    for (int k = 0; k < 2; k++) {
        float fk = float(k);
        float rate = 0.10 + fk * 0.13;
        float hx = w.x * rate + t * 0.0 + fk * 7.3;
        float top = iCamY * 0.25 + 2.2 + fk * 1.1
                  + sin(hx * 0.55) * 0.7 + sin(hx * 1.31 + 1.7) * 0.35;
        if (w.y < top) {
            vec3 hc = mix(vec3(0.13, 0.58, 0.26), palette(iHue + 0.3), 0.08 + fk * 0.06);
            // A lighter crown along the ridge, or a hill is a flat silhouette.
            float crown = smoothstep(0.55, 0.0, top - w.y);
            col = mix(col, hc * (0.62 + fk * 0.30) * (1.0 + crown * 0.55), 0.93);
        }
    }
    // Clouds.
    float cy = w.y - (iCamY + iViewH * 0.72);
    float cx = w.x * 0.28 + 4.0;
    float puff = exp(-pow(cy, 2.0) * 1.6)
               * smoothstep(0.35, 0.9, sin(cx * 0.9) * 0.5 + 0.5 + sin(cx * 2.3) * 0.2);
    col = mix(col, vec3(1.0), clamp(puff, 0.0, 1.0) * 0.55);
    return col;
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float t = mod(iTime, 600.0);
    // World space in tiles: x runs from the camera's tile, y from the ground.
    vec2 w = vec2(iCamFrac + v_uv.x * 15.0, iCamY + v_uv.y * iViewH);

    vec3 col = backdrop(w, t);
    vec3 neon = palette(iHue);

    // ---- the level ----
    float g = groundAt(w.x);
    if (g > (-5.0) && w.y < g) {
        float depth = g - w.y;
        vec3 soil = mix(vec3(0.58, 0.31, 0.16), palette(iHue + 0.62), 0.07);
        // Brick coursing, offset every other row -- the top course is the
        // lighter capping row the originals always had.
        vec2 bc = vec2(w.x, w.y);
        float row = floor(bc.y * 2.0);
        float bx = fract(bc.x * 2.0 + mod(row, 2.0) * 0.5);
        float by = fract(bc.y * 2.0);
        float mortar = smoothstep(0.06, 0.0, min(bx, by))
                     + smoothstep(0.06, 0.0, min(1.0 - bx, 1.0 - by));
        soil *= 0.82 + 0.30 * hash21(vec2(floor(bc.x * 2.0), row));
        soil = mix(soil, soil * 0.55, clamp(mortar, 0.0, 1.0));
        if (depth < 0.5) soil = mix(soil, mix(vec3(0.28, 0.80, 0.28), neon, 0.10), 0.85);
        col = soil * (1.05 + 0.75 * exp(-depth * 0.35));
        // Neon lip along the surface, which is where the pack's look lives.
        col += neon * smoothstep(0.10, 0.0, depth) * (0.5 + iBass * 0.9);
    }

    // ---- pipes ----
    for (int i = 0; i < 4; i++) {
        vec4 P = iPipe[i];
        if (P.x < -50.0) continue;
        float dx = w.x - P.x;
        if (dx < -0.15 || dx > 1.15) continue;
        float top = groundAt(P.x) + P.y;
        if (w.y < top) {
            bool lip = w.y > top - 0.55;
            float half_ = lip ? 0.62 : 0.50;
            if (abs(dx - 0.5) < half_) {
                vec3 pipeCol = mix(vec3(0.10, 0.74, 0.22), palette(iHue + 0.35), 0.08);
                // A highlight band down the left of the barrel gives it round.
                float shade = 0.55 + 0.65 * exp(-pow((dx - 0.30) * 4.0, 2.0));
                col = pipeCol * shade;
                col += neon * smoothstep(0.05, 0.0, abs(w.y - top)) * 1.1;
                col += neon * smoothstep(0.04, 0.0, abs(abs(dx - 0.5) - half_)) * 0.55;
            }
        }
    }

    // ---- blocks ----
    for (int i = 0; i < 10; i++) {
        vec4 B = iBlock[i];
        if (B.x < -50.0) continue;
        vec2 q = w - vec2(B.x + 0.5, B.y + 0.5 + B.w * 0.25);
        if (abs(q.x) > 0.6 || abs(q.y) > 0.6) continue;
        float d = box(q, vec2(0.46));
        if (d > 0.0) continue;
        if (B.z < 0.5) {
            // Brick: four studs and a mortar cross.
            vec3 c = mix(vec3(0.76, 0.36, 0.16), palette(iHue + 0.6), 0.07);
            float cross_ = smoothstep(0.05, 0.0, min(abs(q.x), abs(q.y)));
            col = c * (0.8 + 0.4 * hash21(vec2(B.x, B.y))) * (1.0 - cross_ * 0.45);
        } else if (B.z < 1.5) {
            // Question block: gold, with a mark built from an arc and a dot.
            vec3 c = mix(vec3(1.0, 0.80, 0.14), palette(iHue + 0.12), 0.07);
            col = c * (0.85 + 0.3 * sin(t * 6.0 + B.x));
            float ring = abs(length(q - vec2(0.0, 0.10)) - 0.15) - 0.045;
            float stem = box(q - vec2(0.055, -0.03), vec2(0.045, 0.10));
            float dot_ = length(q - vec2(0.055, -0.24)) - 0.05;
            float mark = min(max(ring, -q.y + 0.02), min(stem, dot_));
            col = mix(col, vec3(0.25, 0.14, 0.02), smoothstep(0.02, 0.0, mark));
        } else {
            col = mix(vec3(0.42, 0.26, 0.12), palette(iHue + 0.6), 0.18);
        }
        col += neon * smoothstep(0.05, 0.0, abs(d)) * (0.5 + iBeat * 0.7);
    }

    // ---- coins ----
    for (int i = 0; i < 10; i++) {
        vec4 C = iCoin[i];
        if (C.z <= 0.0) continue;
        vec2 q = w - C.xy;
        if (dot(q, q) > 0.36) continue;
        // Spin: the disc narrows to an edge and back.
        float wdt = 0.06 + 0.16 * abs(cos(C.w));
        float d = length(q / vec2(wdt, 0.20)) - 1.0;
        vec3 gold = mix(vec3(1.0, 0.85, 0.20), palette(iHue + 0.1), 0.2);
        col = mix(col, gold, smoothstep(0.08, 0.0, d) * C.z);
        col += gold * exp(-max(d, 0.0) * 5.0) * C.z * (0.5 + iTreble * 1.2) * iGlitter;
    }

    // ---- goombas ----
    for (int i = 0; i < 8; i++) {
        vec4 E = iEnemy[i];
        if (E.x < -50.0) continue;
        float squash = clamp(E.z, 0.0, 1.0);
        vec2 q = w - vec2(E.x, E.y + 0.38 * (1.0 - squash));
        if (abs(q.x) > 0.8 || abs(q.y) > 0.8) continue;
        vec2 r = q / vec2(0.42, 0.38 * (1.0 - squash * 0.82) + 0.06);
        float body = length(r) - 1.0;
        vec3 bc = mix(vec3(0.56, 0.30, 0.12), palette(iHue + 0.5), 0.07);
        col = mix(col, bc, smoothstep(0.12, 0.0, body));
        if (squash < 0.5) {
            // Eyes and the angry brow that made them read as cross.
            for (int s = -1; s <= 1; s += 2) {
                vec2 e = q - vec2(0.16 * float(s), 0.10);
                col = mix(col, vec3(1.0), smoothstep(0.02, 0.0, length(e / vec2(0.08, 0.11)) - 1.0));
                col = mix(col, vec3(0.05), smoothstep(0.02, 0.0, length(e - vec2(0.03 * E.w, 0.0)) - 0.035));
                vec2 bq = q - vec2(0.15 * float(s), 0.21);
                col = mix(col, vec3(0.12, 0.05, 0.02),
                          smoothstep(0.02, 0.0, box(bq, vec2(0.10, 0.028))));
            }
        }
        col += neon * smoothstep(0.07, 0.0, abs(body)) * 0.5;
    }

    // ---- the runner ----
    {
        vec2 q = w - vec2(iHero.x, iHero.y);
        if (abs(q.x) < 0.9 && q.y > -0.2 && q.y < 1.4) {
            float f = iHero.z;
            vec3 shirt = mix(vec3(0.90, 0.14, 0.12), palette(iHue + 0.05), 0.15);
            vec3 jeans = mix(vec3(0.16, 0.30, 0.85), palette(iHue + 0.45), 0.15);
            vec3 skin = vec3(1.0, 0.80, 0.62);
            // Legs stride while grounded, tuck while airborne.
            float stride = sin(t * 16.0 + iHero.x * 3.0) * (1.0 - iHeroAir);
            float cover = 0.0;
            float legs = min(box(q - vec2(-0.10 + stride * 0.09, 0.16), vec2(0.09, 0.17)),
                             box(q - vec2(0.10 - stride * 0.09, 0.16), vec2(0.09, 0.17)));
            float m = smoothstep(0.03, 0.0, legs);
            col = mix(col, jeans, m); cover = max(cover, m);
            float torso = box(q - vec2(0.0, 0.52), vec2(0.20, 0.22));
            m = smoothstep(0.03, 0.0, torso);
            col = mix(col, shirt, m); cover = max(cover, m);
            float head = length(q - vec2(0.02 * f, 0.86)) - 0.19;
            m = smoothstep(0.03, 0.0, head);
            col = mix(col, skin, m); cover = max(cover, m);
            // Cap, brim forward, and the moustache.
            float cap = min(box(q - vec2(0.02 * f, 0.99), vec2(0.19, 0.08)),
                            box(q - vec2(0.16 * f, 0.95), vec2(0.12, 0.04)));
            m = smoothstep(0.03, 0.0, cap);
            col = mix(col, shirt, m); cover = max(cover, m);
            float tash = box(q - vec2(0.09 * f, 0.80), vec2(0.10, 0.035));
            col = mix(col, vec3(0.18, 0.09, 0.04), smoothstep(0.02, 0.0, tash));
            // An arm thrown up on the way up sells the jump.
            float arm = box(q - vec2(0.24 * f, 0.62 + iHeroAir * 0.22), vec2(0.07, 0.16));
            m = smoothstep(0.03, 0.0, arm);
            col = mix(col, shirt, m); cover = max(cover, m);
            col += palette(iHue + 0.5) * iHeroStumble * cover * 0.8;
        }
    }

    // Sparkle, and the beat on the whole frame.
    vec2 sp = w * 6.0;
    vec2 sg = floor(sp) + floor(t * 12.0);
    if (hash21(sg) > 0.9965) {
        vec2 sd = fract(sp) - vec2(hash21(sg + 3.1), hash21(sg + 7.7));
        col += vec3(1.0) * exp(-dot(sd, sd) * 26.0) * (0.3 + iTreble * 0.9) * iGlitter;
    }
    col += neon * iBeat * 0.07;
    // Aspect-corrected, or the vignette is an ellipse on a wide screen.
    col *= smoothstep(1.05, 0.30, length((v_uv - 0.5) * vec2(A, 1.0)));
    col = col / (1.0 + col * 0.30);
    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.85)), 1.0);
}
"""


class SuperMario(BaseEffect):
    """Auto-running platformer: real terrain, real gravity, real decisions."""

    NAME = 'Super Mario'
    AUTHOR = 'unicorn-viz'
    TAGS = ['games', 'arcade', 'neon', 'energetic', 'groovy', 'hard']
    PING_PONG_FRIENDS = [
        'Donkey Kong', 'Mario Kart', 'Neon Pac', 'Galaga', 'Breakout',
        'Audio Spectrum',
    ]

    _RUN = 7.0              # tiles / s at rest
    _JUMP_V = 21.0          # tiles / s launch
    _CAM_LEAD = 4.5         # tiles the runner sits behind the camera's left edge

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.2)),
            'glitter': float(self.config.get('glitter', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._sky = float(self.rng.uniform(0.0, 1.0))
        self._seed = float(self.rng.uniform(0.0, 1000.0))

        # Level state. Columns are generated once, on demand, and cached by
        # absolute tile index so the world behind the runner is the world he
        # already ran through.
        self._col_cache: dict[int, float] = {}
        self._feature_cache: dict[int, None] = {}
        self._blocks: list[dict] = []
        self._enemies: list[dict] = []
        self._pipes: list[dict] = []
        self._coins = np.zeros((_NCOIN, 5), dtype=np.float64)   # x y vy life spin

        self._cam = float(self.rng.uniform(0.0, 500.0))
        self._hero_x = self._cam + self._CAM_LEAD
        self._hero_y = float(_GROUND_Y)
        self._hero_vy = 0.0
        self._air = 0.0
        self._coyote = 0.0
        self._stumble = 0.0
        self._cam_y = 0.0
        self._view_h = 11.25

        self._tbl_ground = np.zeros(_COLS, dtype='f4')
        self._tbl_block = np.zeros((_NBLOCK, 4), dtype='f4')
        self._tbl_enemy = np.zeros((_NENEMY, 4), dtype='f4')
        self._tbl_coin = np.zeros((_NCOIN, 4), dtype='f4')
        self._tbl_pipe = np.zeros((4, 4), dtype='f4')
        # Run the level in a little so the first frame is mid-level, never a
        # blank start on an empty plain.
        for _ in range(240):
            self._step(1.0 / 60.0, 1.0)

    # ---------------------------------------------------------------- level
    def _ground(self, tile: int) -> float:
        """Ground top at an absolute tile index, generated once and cached."""
        got = self._col_cache.get(tile)
        if got is not None:
            return got
        r = self._rand(tile, 1)
        h = _GROUND_Y
        # A long, slow terrace so the run is not dead flat.
        h += math.floor(1.6 + 1.6 * math.sin(tile * 0.045 + self._seed * 0.01)) - 1.0
        # Pits are decided per nine-tile block, not per tile: deciding per tile
        # meant the two conditions could only ever agree on a single column, so
        # a "gap" was one missing tile and the runner barely had to jump.
        # Capped at three wide because that is what a jump actually clears --
        # airtime is 2*v/g = 0.68s, which carries him 4.8 tiles at base speed.
        blk = tile // 9
        if self._rand(blk, 11) > 0.62:
            span = 2 + int(self._rand(blk, 12) * 2.0)
            start = 3 + int(self._rand(blk, 13) * 3.0)
            off = tile - blk * 9
            if start <= off < start + span:
                h = _PIT
        self._col_cache[tile] = h
        if len(self._col_cache) > 4000:
            cutoff = int(self._cam) - 80
            for k in [k for k in self._col_cache if k < cutoff]:
                del self._col_cache[k]
        return h

    def _rand(self, tile: int, salt: int) -> float:
        """Deterministic per-tile noise, so a column is the same every time."""
        x = math.sin(tile * 12.9898 + salt * 78.233 + self._seed) * 43758.5453
        return x - math.floor(x)

    def _spawn_features(self, tile: int) -> None:
        """Place blocks, pipes and goombas as a column comes into existence."""
        if tile in self._feature_cache:
            return
        self._feature_cache[tile] = None
        if len(self._feature_cache) > 4000:
            cutoff = int(self._cam) - 80
            for k in [k for k in self._feature_cache if k < cutoff]:
                del self._feature_cache[k]
        g = self._ground(tile)
        if g <= _PIT:
            return
        r = self._rand(tile, 3)
        far_enough = all(abs(q['x'] - tile) > 7.0 for q in self._pipes)
        if r > 0.93 and far_enough and len(self._pipes) < 4:
            self._pipes.append({'x': float(tile), 'h': 1.0 + math.floor(self._rand(tile, 4) * 2.0)})
        elif r > 0.80 and len(self._blocks) < _NBLOCK:
            kind = 1.0 if self._rand(tile, 5) > 0.45 else 0.0
            self._blocks.append({'x': float(tile), 'y': g + 3.5, 'kind': kind, 'hit': 0.0})
        if self._rand(tile, 6) > 0.90 and len(self._enemies) < _NENEMY:
            self._enemies.append({'x': float(tile), 'y': g, 'squash': 0.0, 'face': -1.0})

    # ----------------------------------------------------------------- sim
    def _gap_ahead(self, x: float) -> bool:
        """True when the ground runs out within jumping distance."""
        for d in (1.0, 1.6, 2.2):
            if self._ground(int(math.floor(x + d))) <= _PIT:
                return True
        return False

    def _enemy_ahead(self, x: float) -> bool:
        return any(0.4 < e['x'] - x < 2.0 and e['squash'] < 0.5 for e in self._enemies)

    def _block_ahead(self, x: float, y: float) -> bool:
        """True when an unopened question block is within reach of a jump.

        The apex is v^2/2g above the launch, so a block is only worth going for
        if its underside actually falls inside that -- otherwise he would jump
        at every block he passes and never reach one.
        """
        reach = (self._JUMP_V ** 2) / (2.0 * _GRAVITY)
        return any(b['kind'] == 1.0
                   and 0.8 < b['x'] + 0.5 - x < 2.4
                   and 0.5 < b['y'] - y < reach
                   for b in self._blocks)

    def _step(self, dt: float, tempo: float) -> None:
        """One tick of the level, the runner and everything he interacts with."""
        run = self._RUN * tempo
        self._hero_x += run * dt

        grounded = self._air < 0.5
        gnd = self._ground(int(math.floor(self._hero_x)))
        # Coyote time: a jump asked for just after leaving an edge still fires,
        # which is the difference between a runner and a physics toy.
        self._coyote = 0.08 if grounded else max(0.0, self._coyote - dt)

        want = (self._gap_ahead(self._hero_x)
                or self._enemy_ahead(self._hero_x)
                or self._block_ahead(self._hero_x, self._hero_y))
        if want and self._coyote > 0.0 and self._hero_vy <= 0.0:
            self._hero_vy = self._JUMP_V * (0.85 + self._bass * 0.30)
            self._coyote = 0.0

        self._hero_vy -= _GRAVITY * dt
        prev_y = self._hero_y
        self._hero_y += self._hero_vy * dt

        if gnd > _PIT and self._hero_y <= gnd and self._hero_vy <= 0.0:
            self._hero_y = gnd
            self._hero_vy = 0.0
            self._air = 0.0
        else:
            self._air = 1.0
        if self._hero_y < -3.0:                 # fell in a pit: put him back
            self._hero_y = max(_GROUND_Y, gnd if gnd > _PIT else _GROUND_Y) + 6.0
            self._hero_vy = 0.0
            self._stumble = 1.0

        self._cam = self._hero_x - self._CAM_LEAD
        # Lift the camera only once he climbs past the upper third, and let it
        # down gently. Without it a jump from a high terrace apexes off the top
        # of the frame -- v^2/2g is 3.6 tiles of rise on top of the terrace.
        # Only once he is genuinely near the top of the frame. At 0.62 it
        # triggered on nearly every jump, never settled between them, and the
        # ground spent most of the run pushed off the bottom edge.
        want_y = max(0.0, self._hero_y - self._view_h * 0.82)
        rate = 8.0 if want_y > self._cam_y else 3.0
        self._cam_y += (want_y - self._cam_y) * min(1.0, dt * rate)
        for tile in range(int(self._cam), int(self._cam) + _COLS + 2):
            self._spawn_features(tile)

        # Blocks: opened from below, and they pop a coin when they are.
        for b in self._blocks:
            b['hit'] = max(0.0, b['hit'] - dt * 6.0)
            # Swept against the block's underside. A fixed window cannot work
            # here: he rises 21 tiles/s, which is 0.35 tiles per frame, so any
            # window narrower than a frame's travel is a one-frame target and
            # is usually stepped straight over.
            head_now = self._hero_y + _HEAD
            head_prev = prev_y + _HEAD
            if (b['kind'] == 1.0 and abs(b['x'] + 0.5 - self._hero_x) < 0.75
                    and self._hero_vy > 0.0
                    and head_prev < b['y'] <= head_now):
                b['kind'] = 2.0
                b['hit'] = 1.0
                self._pop_coin(b['x'] + 0.5, b['y'] + 1.0)
        self._blocks = [b for b in self._blocks if b['x'] > self._cam - 6.0]
        self._pipes = [q for q in self._pipes if q['x'] > self._cam - 6.0]

        # Goombas walk, and get stood on.
        for e in self._enemies:
            if e['squash'] >= 1.0:
                continue
            if e['squash'] > 0.0:
                e['squash'] = min(1.0, e['squash'] + dt * 3.0)
                continue
            e['x'] += e['face'] * 2.4 * dt * tempo
            eg = self._ground(int(math.floor(e['x'])))
            if eg <= _PIT:
                e['x'] = -999.0
                continue
            e['y'] = eg
            dx = self._hero_x - e['x']
            if abs(dx) < 0.45 and abs(self._hero_y - e['y']) < 1.0:
                if self._hero_vy < 0.0 and self._hero_y > e['y'] + 0.3:
                    e['squash'] = 0.01
                    self._hero_vy = self._JUMP_V * 0.55     # the stomp bounce
                else:
                    self._stumble = 1.0
        self._enemies = [e for e in self._enemies
                         if e['x'] > self._cam - 6.0 and e['squash'] < 1.0]

        live = self._coins[:, 3] > 0.0
        if live.any():
            self._coins[live, 1] += self._coins[live, 2] * dt
            self._coins[live, 2] -= _GRAVITY * 0.45 * dt
            self._coins[live, 3] -= dt * 0.9
            self._coins[live, 4] += dt * 9.0
        self._stumble = max(0.0, self._stumble - dt * 1.8)

    def _pop_coin(self, x: float, y: float) -> None:
        """Throw a coin up out of an opened block."""
        free = np.where(self._coins[:, 3] <= 0.0)[0]
        if not free.size:
            return
        self._coins[int(free[0])] = (x, y, 13.0, 1.0, float(self.rng.uniform(0.0, 6.0)))

    def update(self, dt: float, audio: AudioData) -> None:
        """Advance the run; the beat buys a jump when one is available."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 1.8))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 1.8))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 1.8))
        if audio.beat > 0.5:
            self._beat = 1.0
            # Not every beat: jumping on all of them kept him in the air and
            # the run cycle never showed.
            if (self._coyote > 0.0 and self._hero_vy <= 0.0
                    and self.rng.random() < 0.28):
                self._hero_vy = self._JUMP_V * (0.9 + self._bass * 0.45)
                self._coyote = 0.0
        self._beat = max(0.0, self._beat - dt * 3.4)

        tempo = max(0.05, float(self.parameters['speed'])) * (0.8 + self._bass * 0.55)
        self._step(min(dt, 1.0 / 30.0), tempo)

    def render(self) -> None:
        """Upload the level and everything standing on it."""
        base = int(math.floor(self._cam))
        for i in range(_COLS):
            self._tbl_ground[i] = self._ground(base + i)

        self._tbl_block[:] = _OFF
        for i, b in enumerate(self._blocks[:_NBLOCK]):
            self._tbl_block[i] = (b['x'] - base, b['y'], b['kind'], b['hit'])
        self._tbl_enemy[:] = _OFF
        for i, e in enumerate(self._enemies[:_NENEMY]):
            self._tbl_enemy[i] = (e['x'] - base, e['y'], e['squash'], e['face'])
        self._tbl_pipe[:] = _OFF
        for i, q in enumerate(self._pipes[:4]):
            self._tbl_pipe[i] = (q['x'] - base, q['h'], 0.0, 0.0)
        self._tbl_coin[:, 0] = self._coins[:, 0] - base
        self._tbl_coin[:, 1] = self._coins[:, 1]
        self._tbl_coin[:, 2] = np.clip(self._coins[:, 3], 0.0, 1.0)
        self._tbl_coin[:, 3] = self._coins[:, 4]

        p = self._prog
        p['iResolution'].value = (float(self.width), float(self.height))
        self._view_h = _VIEW_W / max(float(self.width) / max(float(self.height), 1.0), 0.1)
        p['iTime'].value = float(self.time)
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iHue'].value = self._hue
        p['iGlitter'].value = max(0.0, float(self.parameters['glitter']))
        p['iSky'].value = self._sky
        p['iCamFrac'].value = float(self._cam - base)
        p['iCamY'].value = float(self._cam_y)
        p['iViewH'].value = float(self._view_h)
        p['iGround'].write(self._tbl_ground.tobytes())
        p['iBlock'].write(self._tbl_block.tobytes())
        p['iEnemy'].write(self._tbl_enemy.tobytes())
        p['iCoin'].write(self._tbl_coin.tobytes())
        p['iPipe'].write(self._tbl_pipe.tobytes())
        p['iHero'].value = (float(self._hero_x - base), float(self._hero_y),
                            1.0)
        p['iHeroAir'].value = float(self._air)
        p['iHeroStumble'].value = float(self._stumble)
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()

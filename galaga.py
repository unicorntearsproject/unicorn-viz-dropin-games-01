"""Galaga — a real space-shooter simulation rendered as a neon light-show.

Like :class:`NeonPac`, this is not a procedural "shooter-ish" shader: it runs an
actual arcade space battle on the CPU (the Breakout architecture) and lets a
GLSL shader render the true game state as glowing neon over a cosmic backdrop.

* A **self-playing fighter** at the bottom that slides under enemy columns to
  line up shots, dodges incoming fire, and blasts the swarm — drawn as a glossy
  neon ship with a flickering engine flame.
* An **enemy formation** (8×5 grid, alive/HP stored in a texture like Breakout's
  bricks) that sways side-to-side, drops a rank at each edge, speeds up as it
  thins, and periodically sends **dive-bombers** peeling out in swooping arcs.
* Real **player + enemy bullets** with glowing trails, **explosions** that bloom
  and spark on every kill, and an occasional **bonus mystery saucer** streaking
  across the top.
* **Seven wave formations** (full, columns, diamond, V, checker, arc, boss rank)
  that rotate as each wave is cleared — the shooter analogue of Neon Pac's
  rotating boards.

The whole battle flies up through a warp starfield (streaking stars) laced with
a plasma nebula and comets, and everything drips neon: glossy sprites, bloom,
sparkle-glitter, beat-synced flashes.

Audio reactivity (all bands wired):
  - bass    → game tempo, warp speed, ship/enemy swell, engine thrust
  - mid     → enemy fire rate, palette shimmer
  - treble  → star + spark sparkle-glitter
  - beat    → global flash, palette shift, brief speed surge

Randomised per activation: starting wave, palette hue, formation drift
direction, and ``self.time`` — no two raids look or play alike from frame 0.
Live tweakables: speed, reactivity, glitter.
"""
from __future__ import annotations

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

_FC = 8          # formation columns
_FR = 5          # formation rows
_MAXHP = 3.0     # boss-rank enemies take this many hits
_NBP = 5         # max player bullets in flight
_NBE = 8         # max enemy bullets in flight
_ND = 2          # max simultaneous dive-bombers
_NEX = 6         # max explosions
_NPAT = 7        # wave formation patterns
_OFF = 1000.0    # off-screen marker for inactive sprites

_ROW_COLORS = (
    (1.00, 0.25, 0.90),   # row 0 — magenta (boss rank)
    (1.00, 0.30, 0.25),   # row 1 — red
    (1.00, 0.65, 0.12),   # row 2 — orange
    (0.35, 1.00, 0.45),   # row 3 — green
    (0.25, 0.90, 1.00),   # row 4 — cyan
)

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
uniform float iDeath;
uniform float iHue;
uniform float iGlitter;
uniform sampler2D iEnemy;    // r = hp / MAXHP per cell
uniform vec2  iGrid;         // (cols, rows)
uniform float iFormX;        // formation centre x (world)
uniform float iFormY;        // formation top row y (world)
uniform float iSX;           // column spacing (world)
uniform float iSY;           // row spacing (world)
uniform float iAnim;         // 2-frame leg animation flag
uniform vec3  iRowCol[5];
uniform float iPlayerX;
uniform float iPlayerY;
uniform float iPlayerHit;    // invuln blink
uniform float iThrust;
uniform vec2  iPB[5];        // player bullets (world); x>=900 = inactive
uniform vec2  iEB[8];        // enemy bullets
uniform vec2  iDiver[2];     // dive-bombers
uniform float iDiverAng[2];
uniform vec3  iExpl[6];      // (x, y, age); age<0 = inactive
uniform vec2  iMystery;      // bonus saucer; x>=900 = inactive
in  vec2 v_uv;
out vec4 fragColor;

#define PI  3.14159265
#define TAU 6.28318530
#define MAXHP 3.0

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }

vec3 palette(float t) {
    return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67)));
}

// Warp starfield + plasma nebula + comets — the cosmic backdrop.
vec3 backdrop(vec2 uv, float A, float t) {
    vec2 pp = (uv - 0.5) * vec2(A, 1.0);
    float v = sin(pp.x * 2.3 + t * 0.7) + sin(pp.y * 2.0 - t * 0.6)
            + sin(length(pp) * 3.0 - t);
    vec3 col = palette(v * 0.1 + iHue + iMid * 0.1 + t * 0.02);   // mid shimmers hue
    col *= col;
    col *= 0.18 + 0.10 * iBass;

    for (int L = 0; L < 3; L++) {                 // three warp layers
        float fl = float(L);
        float speed = 0.12 + fl * 0.22 + iBass * 0.35;
        float scale = 30.0 + fl * 42.0;
        vec2 g = vec2(uv.x * A, uv.y + t * speed) * scale;
        vec2 cell = floor(g);
        float h = hash21(cell + fl * 13.0);
        if (h > 0.86) {
            vec2 f = fract(g) - 0.5;
            float streak = smoothstep(0.5, 0.0, length(f * vec2(6.0, 1.0)));
            float tw = 0.6 + 0.4 * sin(t * 3.0 + h * 40.0);
            col += vec3(0.7, 0.85, 1.0) * streak * tw * (0.45 + fl * 0.3)
                 * (0.7 + iTreble * 0.8);
        }
    }
    for (int i = 0; i < 2; i++) {                 // comets
        float fi = float(i);
        float prog = fract(t / (7.0 + fi * 3.0) + fi * 0.5);
        vec2 dir = normalize(vec2(0.8 - 0.5 * fi, -0.6) + 0.001);
        vec2 cp = vec2(-0.9 + fi * 0.5, 0.9) + dir * prog * 2.4;
        vec2 rel = pp - cp;
        float along = dot(rel, dir);
        float perp = length(rel - dir * along);
        float behind = -along;                    // >0 behind the head
        float head = exp(-length(rel) * 34.0);
        float tail = exp(-perp * 55.0) * exp(-max(behind, 0.0) * 3.5)
                   * smoothstep(-0.02, 0.05, behind);
        col += vec3(0.7, 0.85, 1.0) * (head * 2.0 + tail * 0.7);
    }
    return col;
}

// A neon invader bug in cell-local space (q ~ [-0.5,0.5], y up).
vec3 drawInvader(vec2 q, vec3 c, float hp, float anim, float pulse) {
    q /= (1.0 + pulse * 0.12);                          // swell with the music
    float boss = step(1.5, hp);
    float s = mix(0.30, 0.37, boss);
    vec2 bb = abs(q) - vec2(s, s * 0.60);
    float body = length(max(bb, 0.0)) + min(max(bb.x, bb.y), 0.0) - 0.05;
    float m = smoothstep(0.03, -0.02, body);
    vec3 base = c * (1.1 + boss * 0.4);
    vec3 o = base * (0.65 + 0.5 * clamp(q.y + 0.35, 0.0, 1.0)) * m;   // top-lit
    for (int s2 = -1; s2 <= 1; s2 += 2) {                            // eyes
        vec2 ec = vec2(float(s2) * 0.13, 0.03);
        o = mix(o, vec3(1.0), smoothstep(0.08, 0.02, length(q - ec)) * m);
        o = mix(o, vec3(0.05, 0.02, 0.12), smoothstep(0.04, 0.01, length(q - ec)) * m);
    }
    float ant = (smoothstep(0.03, 0.0, abs(q.x - 0.11)) + smoothstep(0.03, 0.0, abs(q.x + 0.11)))
              * smoothstep(0.20, 0.30, q.y) * smoothstep(0.40, 0.30, q.y);
    o += base * ant * 0.8;
    float shift = (anim > 0.5) ? 0.045 : -0.045;                     // animated legs
    float legs = 0.0;
    for (int k = -2; k <= 2; k++) {
        float ly = -s * 0.55 + (((k & 1) == 0) ? shift : -shift);
        legs += smoothstep(0.03, 0.0, length(q - vec2(float(k) * 0.11, ly)));
    }
    o += base * legs * 0.7;
    o += base * m * (0.35 + pulse * 0.5);                            // glow pulses
    o += base * smoothstep(s * 1.7, 0.0, length(q)) * (0.15 + pulse * 0.35);  // aura
    o += vec3(1.0, 0.9, 0.4) * boss
       * smoothstep(0.04, 0.0, abs(length(q) - s * 1.15)) * 0.5;     // boss crown
    return o;
}

// The player fighter in local space (q ~ [-1,1], y up, nose pointing +y).
vec3 drawShip(vec2 q, float t, float thrust, float pulse) {
    q /= (1.0 + pulse * 0.14);                          // breathes with the beat
    vec2 aq = vec2(abs(q.x), q.y);
    // sleek fuselage: a vertical lens pointed at nose and tail
    float prof = sin(clamp((q.y + 0.62) / 1.55, 0.0, 1.0) * PI);
    float fus = smoothstep(0.035, -0.02, abs(q.x) - 0.28 * prof);
    // bold swept-back delta wings
    vec2 wq = rot(0.72) * (aq - vec2(0.30, -0.16));
    float wing = smoothstep(0.04, -0.02, max(abs(wq.x) - 0.46, abs(wq.y) - 0.075));
    float hull = clamp(fus + wing, 0.0, 1.0);
    vec3 body = mix(vec3(0.06, 0.45, 0.95), vec3(0.8, 1.0, 1.0),
                    clamp(q.y * 0.55 + 0.5, 0.0, 1.0));
    vec3 o = body * hull * (1.2 + pulse * 0.7);
    // neon edge outline along the fuselage + wing silhouette
    float edge = smoothstep(0.032, 0.0, abs(abs(q.x) - 0.28 * prof)) * step(0.06, prof)
               + smoothstep(0.03, 0.0, abs(max(abs(wq.x) - 0.46, abs(wq.y) - 0.075)));
    o += vec3(0.4, 1.0, 1.0) * clamp(edge, 0.0, 1.0) * (0.5 + pulse * 0.4);
    // cockpit canopy
    o = mix(o, vec3(1.0, 0.95, 0.45), smoothstep(0.12, 0.0, length((q - vec2(0.0, 0.28)) * vec2(1.4, 1.0))));
    o += vec3(1.0) * smoothstep(0.045, 0.0, length(q - vec2(-0.04, 0.40))) * 0.9;          // spec
    float fx = abs(abs(q.x) - 0.11);                    // twin engine plumes below the tail
    float plume = smoothstep(0.085, 0.0, fx) * smoothstep(-0.5, -0.66, q.y)
                * smoothstep(-1.4, -0.72, q.y) * (0.6 + 0.4 * sin(t * 45.0) + thrust + pulse * 0.6);
    o += mix(vec3(1.0, 0.5, 0.1), vec3(0.3, 0.9, 1.0), 0.5 + 0.5 * sin(t * 30.0)) * max(plume, 0.0);
    o += body * hull * smoothstep(0.95, 0.0, length(q)) * (0.2 + pulse * 0.45);            // aura
    return o;
}

// The bonus mystery saucer in world space around centre c.
vec3 drawUFO(vec2 p, vec2 c, float t, float pulse) {
    vec2 e = (p - c) / (1.0 + pulse * 0.14);
    float disc = 1.0 - smoothstep(-0.012, 0.012, length(e / vec2(0.14, 0.045)) - 1.0);
    float dome = (1.0 - smoothstep(-0.012, 0.012,
                  length((e - vec2(0.0, 0.015)) / vec2(0.062, 0.07)) - 1.0)) * step(0.0, e.y);
    vec3 metal = mix(vec3(0.22, 0.28, 0.42), vec3(0.85, 0.92, 1.0),
                     clamp(e.y * 9.0 + 0.5, 0.0, 1.0));
    vec3 domecol = 0.5 + 0.5 * cos(vec3(0.0, 2.0, 4.0) + t * 3.0);
    vec3 o = metal * disc;
    o = mix(o, domecol, dome * 0.9);
    o += vec3(1.0) * smoothstep(0.028, 0.0, length(e - vec2(-0.045, 0.03))) * 0.6;   // spec
    for (int k = -3; k <= 3; k++) {                     // blinking rim lights
        vec3 lc = 0.5 + 0.5 * cos(vec3(0.0, 2.1, 4.2) + t * 8.0 + float(k) * 1.1);
        o += lc * smoothstep(0.010, 0.0, length(e - vec2(float(k) * 0.037, -0.008)))
           * disc * (0.8 + pulse);
    }
    o += vec3(0.5, 0.9, 1.0) * smoothstep(0.18, 0.0, length(e * vec2(1.0, 2.6))) * (0.12 + pulse * 0.3);
    return o;
}

vec3 drawBullet(vec2 p, vec2 c, float dir, vec3 col) {
    if (c.x > 900.0) return vec3(0.0);
    vec2 d = p - c;
    float behind = -dir * d.y;                 // >0 behind the bolt, <0 ahead
    float core = smoothstep(0.024, 0.0, length(d * vec2(1.8, 0.8)));
    float trail = exp(-abs(d.x) * 110.0) * exp(-max(behind, 0.0) * 16.0)
                * smoothstep(-0.02, 0.02, behind);
    return col * (core * 1.7 + trail * 0.6);
}

vec3 drawExpl(vec2 p, vec3 e) {
    if (e.z < 0.0) return vec3(0.0);
    vec2 d = p - e.xy;
    float r = length(d);
    float age = e.z;
    float R = age * 0.30;
    float ring = smoothstep(0.05, 0.0, abs(r - R)) * (1.0 - age);
    float flash = smoothstep(0.13, 0.0, r) * max(1.0 - age * 2.2, 0.0);
    float spark = pow(0.5 + 0.5 * sin(atan(d.y, d.x) * 9.0), 6.0)
                * smoothstep(R * 1.5, 0.0, r) * (1.0 - age);
    return vec3(1.0, 0.7, 0.25) * (ring * 1.6 + flash * 2.2 + spark * 0.8);
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= A;
    float t = iTime;

    vec3 col = backdrop(v_uv, A, t);
    float pulse = clamp(iBass * 0.7 + iBeat * 0.6, 0.0, 1.3);

    // ---- enemy formation (O(1) grid lookup, Breakout-style) ----
    float fcx = (p.x - iFormX) / iSX + (iGrid.x - 1.0) * 0.5;
    float fcy = (iFormY - p.y) / iSY;
    int cc = int(floor(fcx + 0.5));
    int rr = int(floor(fcy + 0.5));
    if (cc >= 0 && cc < int(iGrid.x) && rr >= 0 && rr < int(iGrid.y)) {
        float hp = texelFetch(iEnemy, ivec2(cc, rr), 0).r * MAXHP;
        if (hp > 0.5) {
            vec2 q = vec2(fcx - float(cc), -(fcy - float(rr)));
            col += drawInvader(q, iRowCol[rr], hp, iAnim, pulse) * (1.0 + iMid * 0.4);
        }
    }

    // ---- dive-bombers ----
    for (int i = 0; i < 2; i++) {
        if (iDiver[i].x > 900.0) continue;
        vec2 q = rot(iDiverAng[i]) * ((p - iDiver[i]) / iSX);
        col += drawInvader(q, vec3(1.0, 0.35, 0.2), 1.0, iAnim, pulse);
    }

    // ---- player ship ----
    {
        float vis = (iPlayerHit > 0.5) ? step(0.5, fract(t * 12.0)) : 1.0;
        vec2 q = (p - vec2(iPlayerX, iPlayerY)) / 0.14;
        col += drawShip(q, t, iThrust, pulse) * vis;
    }

    // ---- bullets ----
    for (int i = 0; i < 5; i++) col += drawBullet(p, iPB[i], 1.0, vec3(0.5, 1.0, 1.0));
    for (int i = 0; i < 8; i++) col += drawBullet(p, iEB[i], -1.0, vec3(1.0, 0.3, 0.5));

    // ---- explosions ----
    for (int i = 0; i < 6; i++) col += drawExpl(p, iExpl[i]);

    // ---- bonus mystery saucer ----
    if (iMystery.x < 900.0) col += drawUFO(p, iMystery, t, pulse);

    // ---- neon post ----
    col += palette(iHue + 0.5) * iFlash * 0.18;
    col += vec3(1.0, 0.1, 0.1) * iDeath * 0.4;
    float sp = hash21(floor(v_uv * iResolution * 0.5) + floor(t * 20.0));
    col += vec3(0.85, 0.92, 1.0) * step(0.994, sp) * (0.35 + iTreble * 0.9) * iGlitter;
    float vig = smoothstep(1.35, 0.25, length(v_uv - 0.5));
    col *= vig;
    col += palette(iHue + 0.2) * iBeat * 0.05 * (1.0 - vig);

    col = col / (1.0 + col * 0.5);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.82));
    fragColor = vec4(col, 1.0);
}
"""


class Galaga(BaseEffect):
    """Self-playing neon space shooter: a real formation battle on the CPU."""

    NAME = 'Galaga'
    AUTHOR = 'unicorn-viz'
    TAGS = ['games', 'arcade', 'shooter', 'energetic', 'groovy', 'neon']
    PING_PONG_FRIENDS = [
        'Breakout', 'Neon Pac',
        'Laser Tunnel', 'Starfield', 'Fireworks', 'Rainbow Trance',
        'Audio Spectrum', 'Audio Centroid',
    ]

    _SX = 0.17
    _SY = 0.16
    _PLAYER_Y = -0.82
    _PB_SPEED = 1.8
    _EB_SPEED = 1.0

    def _init(self) -> None:
        """Build GL resources and start the first wave."""
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.3)),
            'glitter': float(self.config.get('glitter', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._enemy_tex = self.ctx.texture((_FC, _FR), 1, dtype='f4')
        self._enemy_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._enemy_hp = np.zeros((_FR, _FC), dtype=np.float32)
        self._enemy_dirty = True

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._flash = 0.0
        self._death = 0.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._anim = 0.0
        self._anim_timer = 0.0

        self._player_x = 0.0
        self._player_hit = 0.0
        self._thrust = 0.0
        self._fire_cd = 0.0

        self._pb = np.full((_NBP, 2), _OFF, dtype=np.float32)
        self._eb = np.full((_NBE, 2), _OFF, dtype=np.float32)
        self._ex = np.full((_NEX, 3), -1.0, dtype=np.float32)   # x, y, age
        self._divers: list[dict] = []

        self._efire_timer = float(self.rng.uniform(0.4, 1.0))
        self._dive_timer = float(self.rng.uniform(2.0, 4.0))
        self._my_x = _OFF
        self._my_dir = 1.0
        self._my_timer = float(self.rng.uniform(8.0, 16.0))

        self._form_x = 0.0
        self._form_y = 0.62
        self._sway_dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self._wave = int(self.rng.integers(0, _NPAT))
        self._wave_timer = 0.0
        self._spawn_wave(self._wave)

    # ------------------------------------------------------------------ waves
    def _spawn_wave(self, idx: int) -> None:
        """Fill the formation from pattern ``idx`` and reset its position."""
        hp = np.zeros((_FR, _FC), dtype=np.float32)
        cx = (_FC - 1) / 2.0
        cy = (_FR - 1) / 2.0
        pat = idx % _NPAT
        for r in range(_FR):
            for c in range(_FC):
                dc = abs(c - cx)
                if pat == 0:                                  # full grid
                    hp[r, c] = 1.0
                elif pat == 1:                                # alternate columns
                    hp[r, c] = 1.0 if c % 2 == 0 else 0.0
                elif pat == 2:                                # diamond
                    hp[r, c] = 1.0 if dc + abs(r - cy) <= 3.0 else 0.0
                elif pat == 3:                                # V / arrow
                    hp[r, c] = 1.0 if dc <= r + 0.5 else 0.0
                elif pat == 4:                                # checker
                    hp[r, c] = 1.0 if (r + c) % 2 == 0 else 0.0
                elif pat == 5:                                # arc (top-heavy)
                    hp[r, c] = 1.0 if (r < 2 or dc >= 2.0) else 0.0
                else:                                         # boss rank
                    hp[r, c] = 1.0 if (r >= 2 and dc >= 1.0) else 0.0
        if pat in (0, 6):
            hp[0, :] = np.where(hp[0, :] > 0, _MAXHP, 0.0)     # tough top rank
        self._enemy_hp = hp
        self._enemy_dirty = True
        self._form_x = 0.0
        self._form_y = 0.62
        self._sway_dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self._wave_timer = 0.0
        self._pb[:] = _OFF
        self._eb[:] = _OFF
        self._divers.clear()

    # -------------------------------------------------------------- geometry
    def _enemy_x(self, c: int) -> float:
        return self._form_x + (c - (_FC - 1) / 2.0) * self._SX

    def _enemy_y(self, r: int) -> float:
        return self._form_y - r * self._SY

    def _alive_cols(self) -> list[int]:
        return [c for c in range(_FC) if self._enemy_hp[:, c].max() > 0.0]

    # ------------------------------------------------------------------ sim
    def _player_ai(self, dt: float, tempo: float, bound: float) -> None:
        """Slide under enemy columns to shoot; dodge incoming fire."""
        cols = self._alive_cols()
        target = self._player_x
        if cols:
            target = min((self._enemy_x(c) for c in cols),
                         key=lambda x: abs(x - self._player_x))
        # dodge close descending enemy bullets
        for k in range(_NBE):
            bx, by = self._eb[k]
            if bx < 900.0 and by < self._PLAYER_Y + 0.4 and abs(bx - self._player_x) < 0.16:
                target = self._player_x + (0.35 if bx < self._player_x else -0.35)
        self._player_x += float(np.clip(target - self._player_x, -1.0, 1.0)) \
            * min(1.0, dt * 6.0 * tempo)
        self._player_x = float(np.clip(self._player_x, -bound, bound))

        self._fire_cd = max(0.0, self._fire_cd - dt)
        aligned = any(self._enemy_hp[:, c].max() > 0.0
                      and abs(self._enemy_x(c) - self._player_x) < 0.05 for c in range(_FC))
        if self._fire_cd <= 0.0 and (aligned or not cols):
            self._fire_player()
            self._fire_cd = float(self.rng.uniform(0.28, 0.5))
            self._thrust = 0.6

    def _fire_player(self) -> None:
        free = np.where(self._pb[:, 0] > 900.0)[0]
        if free.size:
            self._pb[free[0]] = (self._player_x, self._PLAYER_Y + 0.08)

    def _enemy_fire(self) -> None:
        cols = self._alive_cols()
        if not cols:
            return
        c = int(self.rng.choice(cols))
        rows = np.where(self._enemy_hp[:, c] > 0.0)[0]
        r = int(rows.max())                                   # bottom-most in column
        free = np.where(self._eb[:, 0] > 900.0)[0]
        if free.size:
            self._eb[free[0]] = (self._enemy_x(c), self._enemy_y(r) - 0.05)

    def _start_dive(self) -> None:
        if len(self._divers) >= _ND:
            return
        alive = [(r, c) for r in range(_FR) for c in range(_FC) if self._enemy_hp[r, c] > 0.0]
        if not alive:
            return
        r, c = alive[int(self.rng.integers(0, len(alive)))]
        self._divers.append({
            'r': r, 'c': c, 'phase': 0.0, 'hp': float(self._enemy_hp[r, c]),
            'x0': self._enemy_x(c), 'y0': self._enemy_y(r),
            'tx': self._player_x, 'x': self._enemy_x(c), 'y': self._enemy_y(r),
            'ang': 0.0, 'sweep': float(self.rng.uniform(0.2, 0.45)) * (1 if self.rng.random() < 0.5 else -1),
        })
        self._enemy_hp[r, c] = 0.0                            # hidden from grid while diving
        self._enemy_dirty = True

    def _update_divers(self, dt: float, tempo: float) -> None:
        for d in list(self._divers):
            d['phase'] += dt * 0.5 * tempo
            ph = d['phase']
            ny = d['y0'] + (-1.15 - d['y0']) * (ph * ph)      # accelerate downward
            nx = d['x0'] + (d['tx'] - d['x0']) * ph + d['sweep'] * np.sin(ph * 6.0) * (1.0 - ph)
            d['ang'] = float(np.arctan2(ny - d['y'], nx - d['x']) + np.pi / 2.0)
            d['x'], d['y'] = float(nx), float(ny)
            if self.rng.random() < dt * 1.5:                  # divers strafe
                free = np.where(self._eb[:, 0] > 900.0)[0]
                if free.size:
                    self._eb[free[0]] = (d['x'], d['y'] - 0.05)
            if ph >= 1.0:                                     # survived the run
                self._enemy_hp[d['r'], d['c']] = d['hp']
                self._enemy_dirty = True
                self._divers.remove(d)

    def _update_bullets(self, dt: float, tempo: float) -> None:
        pa = self._pb[:, 0] < 900.0
        self._pb[pa, 1] += self._PB_SPEED * dt * tempo
        self._pb[self._pb[:, 1] > 1.08] = _OFF
        ea = self._eb[:, 0] < 900.0
        self._eb[ea, 1] -= self._EB_SPEED * dt * tempo
        self._eb[self._eb[:, 1] < -1.08] = _OFF

    def _spawn_explosion(self, x: float, y: float) -> None:
        free = np.where(self._ex[:, 2] < 0.0)[0]
        slot = free[0] if free.size else int(np.argmax(self._ex[:, 2]))
        self._ex[slot] = (x, y, 0.0)

    def _collisions(self) -> None:
        # player bullets vs formation + divers + mystery
        for k in range(_NBP):
            bx, by = self._pb[k]
            if bx > 900.0:
                continue
            fcx = (bx - self._form_x) / self._SX + (_FC - 1) / 2.0
            fcy = (self._form_y - by) / self._SY
            c = int(round(fcx))
            r = int(round(fcy))
            if 0 <= c < _FC and 0 <= r < _FR and self._enemy_hp[r, c] > 0.0:
                self._enemy_hp[r, c] -= 1.0
                self._enemy_dirty = True
                self._pb[k] = _OFF
                if self._enemy_hp[r, c] <= 0.0:
                    self._spawn_explosion(self._enemy_x(c), self._enemy_y(r))
                    self._flash = max(self._flash, 0.35)
                continue
            for d in list(self._divers):
                if abs(bx - d['x']) < 0.07 and abs(by - d['y']) < 0.07:
                    d['hp'] -= 1.0
                    self._pb[k] = _OFF
                    if d['hp'] <= 0.0:
                        self._spawn_explosion(d['x'], d['y'])
                        self._flash = max(self._flash, 0.4)
                        self._divers.remove(d)
                    break
            else:
                if self._my_x < 900.0 and abs(bx - self._my_x) < 0.08 and by > 0.82:
                    self._spawn_explosion(self._my_x, 0.86)
                    self._flash = max(self._flash, 0.8)
                    self._my_x = _OFF
                    self._pb[k] = _OFF

        if self._player_hit > 0.0:
            return
        px = self._player_x
        for k in range(_NBE):                                 # enemy bullets vs player
            bx, by = self._eb[k]
            if bx < 900.0 and abs(bx - px) < 0.08 and abs(by - self._PLAYER_Y) < 0.08:
                self._hit_player()
                return
        for d in self._divers:                                # divers vs player
            if abs(d['x'] - px) < 0.09 and abs(d['y'] - self._PLAYER_Y) < 0.09:
                self._hit_player()
                return

    def _hit_player(self) -> None:
        self._death = 1.0
        self._player_hit = 1.4
        self._flash = max(self._flash, 0.6)
        self._eb[:] = _OFF
        self._spawn_explosion(self._player_x, self._PLAYER_Y)

    def _update_formation(self, dt: float, tempo: float, bound: float) -> None:
        cols = self._alive_cols()
        if not cols:
            return
        alive_frac = float((self._enemy_hp > 0.0).sum()) / (_FC * _FR)
        speed = 0.18 * (1.0 + (1.0 - alive_frac) * 2.2) * tempo
        self._form_x += self._sway_dir * speed * dt
        left = self._enemy_x(min(cols)) - self._SX * 0.5
        right = self._enemy_x(max(cols)) + self._SX * 0.5
        if right > bound:
            self._form_x -= right - bound
            self._sway_dir = -1.0
            self._form_y -= 0.05
        elif left < -bound:
            self._form_x += -bound - left
            self._sway_dir = 1.0
            self._form_y -= 0.05
        if self._form_y < self._PLAYER_Y + 0.35:              # swarm reached the ship
            self._death = 1.0
            self._flash = max(self._flash, 0.8)
            self._advance_wave()

    def _update_mystery(self, dt: float) -> None:
        if self._my_x < 900.0:
            self._my_x += self._my_dir * dt * 0.5
            if abs(self._my_x) > 1.4:
                self._my_x = _OFF
            return
        self._my_timer -= dt
        if self._my_timer <= 0.0:
            self._my_dir = 1.0 if self.rng.random() < 0.5 else -1.0
            self._my_x = -1.3 * self._my_dir
            self._my_timer = float(self.rng.uniform(12.0, 22.0))

    def _advance_wave(self) -> None:
        self._wave += 1
        self._hue = float((self._hue + self.rng.uniform(0.12, 0.3)) % 1.0)
        self._spawn_wave(self._wave)

    # ---------------------------------------------------------------- update
    def update(self, dt: float, audio: AudioData) -> None:
        """Advance audio envelopes and the whole battle simulation."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 2.0))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 2.0))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 2.0))
        if audio.beat > 0.5:
            self._beat = 1.0
            self._flash = max(self._flash, 0.35)
            self._hue = float((self._hue + 0.02 + self._mid * 0.02) % 1.0)
        self._beat = max(0.0, self._beat - dt * 2.0)
        self._flash = max(0.0, self._flash - dt * 1.8)
        self._death = max(0.0, self._death - dt * 1.6)
        self._player_hit = max(0.0, self._player_hit - dt)
        self._thrust = max(0.0, self._thrust - dt * 3.0)

        self._anim_timer += dt
        if self._anim_timer > 0.35:
            self._anim_timer = 0.0
            self._anim = 0.0 if self._anim > 0.5 else 1.0

        # explosions age out
        act = self._ex[:, 2] >= 0.0
        self._ex[act, 2] += dt * 2.0
        self._ex[self._ex[:, 2] > 1.0] = -1.0

        self._update_mystery(dt)

        tempo = max(0.05, float(self.parameters['speed'])) * (1.0 + self._bass * 0.4)
        bound = self._A * 0.9

        self._update_bullets(dt, tempo)
        self._update_formation(dt, tempo, bound)
        self._update_divers(dt, tempo)
        self._player_ai(dt, tempo, bound)

        self._efire_timer -= dt
        if self._efire_timer <= 0.0:
            self._enemy_fire()
            self._efire_timer = float(self.rng.uniform(0.35, 0.9)) / (1.0 + self._mid * 0.8)
        self._dive_timer -= dt
        if self._dive_timer <= 0.0:
            self._start_dive()
            self._dive_timer = float(self.rng.uniform(2.5, 5.0))

        self._collisions()

        # wave clear → next formation
        self._wave_timer += dt
        if (self._enemy_hp > 0.0).sum() == 0 and not self._divers:
            self._flash = max(self._flash, 0.7)
            self._advance_wave()

    @property
    def _A(self) -> float:
        return float(self.width) / max(float(self.height), 1.0)

    # ---------------------------------------------------------------- render
    def render(self) -> None:
        """Upload the battle state and draw the neon arena."""
        if self._enemy_dirty:
            self._enemy_tex.write((self._enemy_hp / _MAXHP).astype('f4').tobytes())
            self._enemy_dirty = False
        self._enemy_tex.use(location=0)

        p = self._prog
        p['iTime'].value = float(self.time)
        p['iResolution'].value = (float(self.width), float(self.height))
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iFlash'].value = self._flash
        p['iDeath'].value = self._death
        p['iHue'].value = self._hue
        p['iGlitter'].value = max(0.0, float(self.parameters['glitter']))
        p['iEnemy'].value = 0
        p['iGrid'].value = (float(_FC), float(_FR))
        p['iFormX'].value = float(self._form_x)
        p['iFormY'].value = float(self._form_y)
        p['iSX'].value = float(self._SX)
        p['iSY'].value = float(self._SY)
        p['iAnim'].value = float(self._anim)
        p['iRowCol'].value = list(_ROW_COLORS)
        p['iPlayerX'].value = float(self._player_x)
        p['iPlayerY'].value = float(self._PLAYER_Y)
        p['iPlayerHit'].value = 1.0 if self._player_hit > 0.0 else 0.0
        p['iThrust'].value = float(self._thrust)
        p['iPB'].value = [tuple(map(float, b)) for b in self._pb]
        p['iEB'].value = [tuple(map(float, b)) for b in self._eb]

        divers = [(_OFF, _OFF), (_OFF, _OFF)]
        angs = [0.0, 0.0]
        for i, d in enumerate(self._divers[:_ND]):
            divers[i] = (float(d['x']), float(d['y']))
            angs[i] = float(d['ang'])
        p['iDiver'].value = divers
        p['iDiverAng'].value = angs
        p['iExpl'].value = [tuple(map(float, e)) for e in self._ex]
        p['iMystery'].value = (float(self._my_x), 0.86)

        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
        self._enemy_tex.release()

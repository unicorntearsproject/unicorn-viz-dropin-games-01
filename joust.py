"""Joust — a real flap-physics jousting arena rendered as a neon light-show.

Like the other games in this pack, this is not a procedural "joust-ish" shader:
it runs an actual Joust match on the CPU (the Breakout architecture) and lets a
GLSL shader render the true game state as glowing neon over a molten arena.

* A **self-playing knight** on a flying mount that flaps to gain altitude,
  fights gravity, wraps around the screen edges, and dives to joust — drawn as
  a neon bird with animated flapping wings, a rider, and a gleaming lance.
* **Enemy buzzard-riders** with real flap AI that chase the player and try to
  get *above* them (altitude decides a joust).  On a clash the higher rider
  wins; the loser bursts into an explosion and drops an **egg**.
* **Eggs** fall, rest on platforms, and hatch into fresh enemies after a few
  seconds — the player can swoop in to collect them first.  Clearing the arena
  spawns the next (bigger) wave.
* **Neon platforms**, a **molten lava floor** that bubbles, glows, and surges
  on the bass with rising embers, joust flashes, and feather sparkle.

Audio reactivity (all bands wired):
  - bass    → lava surge/glow, flap power, arena pulse
  - mid     → enemy aggression, wing-beat rate, palette shimmer
  - treble  → ember + spark sparkle-glitter
  - beat    → global flash, palette shift, lava jump

Randomised per activation: starting wave size, palette hue, spawn positions and
facings, and ``self.time`` — no two matches look or play alike.  Live
tweakables: speed, reactivity, glitter.
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

_NB = 6          # bird slots (index 0 = player, 1..5 = enemies)
_NEGG = 4        # max eggs
_NPLAT = 5       # platforms
_NEX = 6         # explosions
_OFF = 1000.0

_LAVA_Y = -0.78
_GRAV = 1.15
_FLAP = 0.62
_MAX_VY = 1.25
_CRUISE = 0.85

# Platforms as (x-fraction-of-aspect, world-y, half-width).
_PLATFORMS = [
    (0.00, -0.58, 0.30),
    (-0.70, -0.18, 0.24),
    (0.70, -0.18, 0.24),
    (0.00, 0.30, 0.28),
    (-0.66, 0.46, 0.18),
]

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
uniform float iLavaGlow;
uniform float iPlayerInvuln;
uniform vec2  iBirdPos[6];
uniform float iBirdFace[6];
uniform float iBirdWing[6];
uniform float iBirdKind[6];   // <0 inactive, 0 player, 1 red, 2 purple
uniform vec2  iEgg[4];        // x>=900 inactive
uniform float iEggHatch[4];
uniform vec4  iPlat[5];       // (x, y, half-width, active)
uniform vec3  iExpl[6];       // (x, y, age); age<0 inactive
in  vec2 v_uv;
out vec4 fragColor;

#define PI  3.14159265
#define TAU 6.28318530
#define LAVA_Y -0.78

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }
float cap(vec2 p, vec2 a, vec2 b, float r) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}
float rbox(vec2 p, vec2 b, float r) {
    vec2 d = abs(p) - b + r;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - r;
}

// Dark cavern gradient + faint stars + rising embers.
vec3 backdrop(vec2 uv, float A, float t) {
    vec3 col = mix(vec3(0.05, 0.02, 0.05), vec3(0.02, 0.02, 0.06), uv.y);
    col += palette(iHue + 0.6 + iMid * 0.25) * (0.03 + iMid * 0.04) * (1.0 - uv.y);  // mid shimmer
    vec2 sg = uv * vec2(A, 1.0) * 60.0;
    float sh = hash21(floor(sg));
    if (sh > 0.95) {
        float tw = 0.5 + 0.5 * sin(t * 2.0 + sh * 40.0);
        col += vec3(0.6, 0.7, 0.9) * smoothstep(0.4, 0.0, length(fract(sg) - 0.5)) * tw * 0.5;
    }
    vec2 p = (uv - 0.5) * vec2(A, 1.0);
    vec2 eg = vec2(p.x * 9.0, (p.y - t * 0.35) * 9.0);
    float eh = hash21(floor(eg));
    if (eh > 0.93 && p.y > LAVA_Y) {
        float rise = smoothstep(0.7, -0.8, p.y);
        col += vec3(1.0, 0.5, 0.12) * smoothstep(0.42, 0.0, length(fract(eg) - 0.5))
             * rise * (0.5 + iTreble * 0.8);
    }
    return col;
}

vec3 lava(vec2 p, float t) {
    float surf = LAVA_Y + 0.02 * sin(p.x * 7.0 + t * 2.0)
               + 0.014 * sin(p.x * 13.0 - t * 3.0) + iLavaGlow * 0.035;
    vec3 o = vec3(0.0);
    if (p.y < surf) {
        float n = 0.5 + 0.5 * sin(p.x * 18.0 + t * 4.0 + sin(p.y * 15.0 + t));
        vec3 molten = mix(vec3(0.9, 0.18, 0.02), vec3(1.0, 0.62, 0.12), n);
        o = molten * (0.6 + 0.4 * n) * (0.75 + iLavaGlow * 0.7);
        float b = hash21(floor(vec2(p.x * 22.0, t * 2.5)));
        o += vec3(1.0, 0.75, 0.25) * step(0.97, b) * 0.6;
    }
    float g = exp(-max(p.y - surf, 0.0) * 6.0);
    o += vec3(1.0, 0.42, 0.06) * g * (0.4 + iLavaGlow * 0.8 + iBass * 0.4);
    return o;
}

// A neon jouster (bird + rider + lance) in local space, facing +x.
vec3 drawBird(vec2 q, float wing, vec3 bc, float t) {
    vec3 o = vec3(0.0);
    // wing (behind body), sweeps with the flap value
    vec2 wp = rot(-0.35 - wing * 0.8) * (q - vec2(-0.10, 0.05));
    float wingSdf = length((wp - vec2(-0.34, 0.0)) / vec2(0.40, 0.15)) - 1.0;
    float wm = smoothstep(0.05, -0.02, wingSdf);
    o = mix(o, bc * 0.8, wm);
    o += bc * smoothstep(0.03, 0.0, abs(wingSdf + 0.15)) * wm * 0.5;   // feather line
    // body
    float bm = smoothstep(0.04, -0.02, length(q / vec2(0.5, 0.26)) - 1.0);
    vec3 shade = bc * (0.68 + 0.5 * clamp(q.y + 0.3, 0.0, 1.0));
    o = mix(o, shade, bm);
    // neck + head
    float nm = smoothstep(0.03, -0.02, cap(q, vec2(0.30, 0.08), vec2(0.62, 0.42), 0.10));
    o = mix(o, shade, nm);
    float hm = smoothstep(0.03, -0.02, length(q - vec2(0.66, 0.46)) - 0.13);
    o = mix(o, bc * 1.05, hm);
    // beak
    vec2 bq = rot(-0.2) * (q - vec2(0.83, 0.45));
    o = mix(o, vec3(1.0, 0.8, 0.2), smoothstep(0.03, -0.02, length(bq / vec2(0.14, 0.05)) - 1.0));
    // eye
    o = mix(o, vec3(0.04, 0.02, 0.06), smoothstep(0.04, 0.0, length(q - vec2(0.67, 0.52))));
    o += vec3(1.0) * smoothstep(0.016, 0.0, length(q - vec2(0.69, 0.55))) * 0.9;
    // legs
    float lm = smoothstep(0.02, -0.01, min(cap(q, vec2(0.05, -0.24), vec2(0.10, -0.52), 0.025),
                                           cap(q, vec2(-0.10, -0.24), vec2(-0.16, -0.52), 0.025)));
    o = mix(o, bc * 0.7, lm);
    // rider (armoured knight)
    vec3 armor = vec3(0.72, 0.77, 0.88);
    float rm = smoothstep(0.03, -0.02, length((q - vec2(-0.02, 0.42)) / vec2(0.16, 0.22)) - 1.0);
    o = mix(o, armor, rm);
    o = mix(o, armor * 1.1, smoothstep(0.03, -0.02, length(q - vec2(0.04, 0.62)) - 0.10));
    o += vec3(1.0) * smoothstep(0.05, 0.0, length(q - vec2(0.0, 0.5))) * rm * 0.4;
    // lance
    float lance = cap(q, vec2(0.10, 0.46), vec2(1.15, 0.30), 0.02);
    o = mix(o, vec3(0.9, 0.95, 1.0), smoothstep(0.02, -0.01, lance));
    o += vec3(1.0, 1.0, 0.8) * smoothstep(0.06, 0.0, length(q - vec2(1.15, 0.30)))
       * (0.6 + 0.4 * sin(t * 10.0));                                  // lance-tip gleam
    // neon aura
    float sil = max(max(bm, nm), max(wm, hm));
    o += bc * sil * 0.35;
    o += bc * smoothstep(0.95, 0.0, length(q)) * 0.12;
    return o;
}

vec3 drawExpl(vec2 p, vec3 e) {
    if (e.z < 0.0) return vec3(0.0);
    vec2 d = p - e.xy;
    float r = length(d);
    if (r > 0.5) return vec3(0.0);
    float age = e.z;
    float R = age * 0.30;
    float ring = smoothstep(0.05, 0.0, abs(r - R)) * (1.0 - age);
    float flash = smoothstep(0.12, 0.0, r) * max(1.0 - age * 2.2, 0.0);
    float feather = pow(0.5 + 0.5 * sin(atan(d.y, d.x) * 7.0), 5.0)
                  * smoothstep(R * 1.5, 0.0, r) * (1.0 - age);
    return vec3(1.0, 0.85, 0.5) * (ring * 1.5 + flash * 2.0 + feather * 0.7);
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= A;
    float t = iTime;
    float wrap = 2.0 * A;

    vec3 col = backdrop(v_uv, A, t);
    col += lava(p, t);

    // platforms
    for (int i = 0; i < 5; i++) {
        if (iPlat[i].w < 0.5) continue;
        vec2 d = p - iPlat[i].xy;
        d.x -= wrap * floor(d.x / wrap + 0.5);
        if (abs(d.x) > iPlat[i].z + 0.08 || abs(d.y) > 0.14) continue;
        float bar = rbox(d, vec2(iPlat[i].z, 0.035), 0.02);
        col = mix(col, vec3(0.03, 0.06, 0.14), smoothstep(0.012, -0.012, bar));
        float topedge = smoothstep(0.02, 0.0, abs(d.y - 0.035)) * step(abs(d.x), iPlat[i].z);
        col += vec3(0.2, 0.7, 1.0) * topedge * (1.0 + iBass * 0.5);
        col += vec3(0.15, 0.5, 1.0) * smoothstep(0.03, 0.0, abs(bar)) * 0.4;
    }

    // eggs
    for (int i = 0; i < 4; i++) {
        if (iEgg[i].x > 900.0) continue;
        vec2 d = p - iEgg[i];
        d.x -= wrap * floor(d.x / wrap + 0.5);
        if (length(d) > 0.16) continue;
        float m = smoothstep(0.02, -0.02, length(d / vec2(0.06, 0.08)) - 1.0);
        float h = iEggHatch[i];
        vec3 ec = mix(vec3(1.0, 0.95, 0.8), palette(iHue + 0.3), 0.3 + h * 0.4);
        col = mix(col, ec, m);
        col += vec3(1.0) * smoothstep(0.03, 0.0, length(d - vec2(-0.02, 0.03))) * 0.7;
        col += ec * m * (0.2 + h * 0.9 * (0.5 + 0.5 * sin(t * (6.0 + h * 20.0))));  // hatch throb
        col += ec * smoothstep(0.14, 0.0, length(d)) * (0.1 + h * 0.35);
    }

    // birds (player at index 0)
    for (int i = 0; i < 6; i++) {
        if (iBirdKind[i] < 0.0) continue;
        vec2 d = vec2(p.x - iBirdPos[i].x, p.y - iBirdPos[i].y);
        d.x -= wrap * floor(d.x / wrap + 0.5);
        if (length(d) > 0.34) continue;
        vec2 q = d / 0.14;
        q.x *= iBirdFace[i];
        vec3 bc = (iBirdKind[i] < 0.5) ? vec3(0.2, 0.9, 1.0)
                : (iBirdKind[i] < 1.5) ? vec3(1.0, 0.3, 0.22)
                                       : vec3(0.85, 0.35, 1.0);
        float vis = (i == 0 && iPlayerInvuln > 0.5) ? step(0.5, fract(t * 12.0)) : 1.0;
        col += drawBird(q, iBirdWing[i], bc, t) * vis;
    }

    // explosions
    for (int i = 0; i < 6; i++) col += drawExpl(p, iExpl[i]);

    // neon post
    col += palette(iHue + 0.5) * iFlash * 0.2;
    float sp = hash21(floor(v_uv * iResolution * 0.5) + floor(t * 20.0));
    col += vec3(0.9, 0.85, 1.0) * step(0.994, sp) * (0.35 + iTreble * 0.9) * iGlitter;
    float vig = smoothstep(1.4, 0.3, length(v_uv - 0.5));
    col *= vig;
    col += palette(iHue + 0.1) * iBeat * 0.05 * (1.0 - vig);

    col = col / (1.0 + col * 0.5);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.82));
    fragColor = vec4(col, 1.0);
}
"""


class _Bird:
    """A flap-physics rider (the player or an enemy)."""

    __slots__ = ('x', 'y', 'vx', 'vy', 'face', 'wing', 'flap_hot', 'kind', 'land_cd')

    def __init__(self, x: float, y: float, kind: int, face: float = 1.0) -> None:
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.face = face
        self.wing = 0.0
        self.flap_hot = 0.0     # recent-flap envelope (drives wing-beat rate)
        self.kind = kind        # 0 player, 1 red enemy, 2 purple enemy
        self.land_cd = 0.0


class Joust(BaseEffect):
    """Self-playing neon Joust: real flap physics and altitude duels on the CPU."""

    NAME = 'Joust'
    AUTHOR = 'unicorn-viz'
    TAGS = ['games', 'arcade', 'neon', 'medieval', 'energetic', 'intense']
    PING_PONG_FRIENDS = [
        'Breakout', 'Neon Pac', 'Galaga',
        'Reactor Breach', 'Laser Tunnel', 'Fireworks', 'Cathedral of Bass',
        'Audio Spectrum', 'Audio Waveforms',
    ]

    def _init(self) -> None:
        """Build GL resources and start the first wave."""
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.3)),
            'glitter': float(self.config.get('glitter', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._flash = 0.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._lava = 0.0

        self._player = _Bird(0.0, 0.4, 0, 1.0)
        self._player_invuln = 1.5
        self._enemies: list[_Bird] = []
        self._eggs: list[dict] = []
        self._ex = np.full((_NEX, 3), -1.0, dtype=np.float32)
        self._wave = int(self.rng.integers(0, 3))
        self._eflap_timer = 0.0
        self._spawn_wave(3 + self._wave)

    @property
    def _A(self) -> float:
        return float(self.width) / max(float(self.height), 1.0)

    # ------------------------------------------------------------------ setup
    def _spawn_wave(self, n: int) -> None:
        """Fill the arena with ``n`` fresh enemies at random edges/altitudes."""
        n = min(n, _NB - 1)
        a = self._A
        for _ in range(n):
            side = -1.0 if self.rng.random() < 0.5 else 1.0
            b = _Bird(side * a * 0.85,
                      float(self.rng.uniform(0.0, 0.6)),
                      int(self.rng.integers(1, 3)),
                      -side)
            b.vx = -side * 0.3
            self._enemies.append(b)

    def _plat_world(self):
        """Yield platforms as (x, y, half-width) in world units."""
        a = self._A
        for fx, y, fw in _PLATFORMS:
            yield fx * a * 0.9, y, fw

    # -------------------------------------------------------------- physics/AI
    def _wrapdx(self, tx: float, x: float) -> float:
        a = self._A
        dx = tx - x
        if dx > a:
            dx -= 2.0 * a
        elif dx < -a:
            dx += 2.0 * a
        return dx

    def _flap(self, b: _Bird) -> None:
        b.vy = min(b.vy + _FLAP, _MAX_VY)
        b.flap_hot = 1.0

    def _ai(self, b: _Bird, dt: float, tempo: float) -> None:
        """Pick a horizontal target and decide when to flap."""
        if b.kind == 0:                                  # player: hunt the lowest enemy
            if self._enemies:
                tgt = min(self._enemies, key=lambda e: abs(self._wrapdx(e.x, b.x)) + abs(e.y - b.y))
                tx, ty = tgt.x, tgt.y + 0.16             # get above the target
            elif self._eggs:                             # otherwise collect an egg
                e = min(self._eggs, key=lambda e: abs(self._wrapdx(e['x'], b.x)))
                tx, ty = e['x'], e['y'] + 0.05
            else:
                tx, ty = b.x, 0.35
        else:                                            # enemy: get above the player
            tx, ty = self._player.x, self._player.y + 0.14

        dx = self._wrapdx(tx, b.x)
        b.face = 1.0 if dx >= 0.0 else -1.0
        desired_vx = math.copysign(_CRUISE * tempo, dx) if abs(dx) > 0.04 else 0.0
        b.vx += (desired_vx - b.vx) * min(1.0, dt * 3.0)

        # flap to reach/hold target altitude (and never sink into the lava)
        near_lava = b.y < _LAVA_Y + 0.22
        want_up = b.y < ty - 0.03 or b.vy < -0.25
        rate = dt * (2.4 + self._mid * 2.0) * tempo
        if near_lava or (want_up and self.rng.random() < rate):
            self._flap(b)

    def _physics(self, b: _Bird, dt: float) -> None:
        b.vy = max(b.vy - _GRAV * dt, -_MAX_VY)
        b.x += b.vx * dt
        b.y += b.vy * dt
        a = self._A
        if b.x > a:
            b.x -= 2.0 * a
        elif b.x < -a:
            b.x += 2.0 * a
        if b.y > 0.95:
            b.y = 0.95
            b.vy = min(b.vy, 0.0)
        # perch on platforms (soft landing from above)
        b.land_cd = max(0.0, b.land_cd - dt)
        if b.land_cd <= 0.0 and b.vy < 0.0:
            for px, py, hw in self._plat_world():
                if abs(self._wrapdx(px, b.x)) < hw and py + 0.02 < b.y < py + 0.14:
                    b.y = py + 0.11
                    b.vy = 0.12
                    b.land_cd = 0.3
                    break
        # wing beat animation
        b.flap_hot = max(0.0, b.flap_hot - dt * 2.5)
        b.wing = math.sin(self.time * (5.0 + b.flap_hot * 8.0 + self._mid * 3.0)
                          + b.x * 3.0)

    def _spawn_explosion(self, x: float, y: float) -> None:
        free = np.where(self._ex[:, 2] < 0.0)[0]
        slot = int(free[0]) if free.size else int(np.argmax(self._ex[:, 2]))
        self._ex[slot] = (x, y, 0.0)

    def _defeat_enemy(self, e: _Bird, drop_egg: bool) -> None:
        self._spawn_explosion(e.x, e.y)
        self._flash = max(self._flash, 0.6)
        if e in self._enemies:
            self._enemies.remove(e)
        if drop_egg and len(self._eggs) < _NEGG:
            self._eggs.append({'x': e.x, 'y': e.y, 'vy': e.vy, 'hatch': 5.0, 'kind': e.kind})

    def _hit_player(self) -> None:
        if self._player_invuln > 0.0:
            return
        self._spawn_explosion(self._player.x, self._player.y)
        self._flash = max(self._flash, 0.7)
        self._player.x = 0.0
        self._player.y = 0.45
        self._player.vx = self._player.vy = 0.0
        self._player_invuln = 1.8

    def _jousts(self) -> None:
        pl = self._player
        for e in list(self._enemies):
            if abs(self._wrapdx(e.x, pl.x)) < 0.13 and abs(e.y - pl.y) < 0.13:
                if pl.y > e.y + 0.02:
                    self._defeat_enemy(e, self.rng.random() < 0.7)
                elif e.y > pl.y + 0.02:
                    self._hit_player()
                    return
                else:                                    # level clash: bounce apart
                    d = math.copysign(0.4, self._wrapdx(pl.x, e.x))
                    pl.vx = d
                    e.vx = -d
                    pl.vy += 0.15
                    e.vy += 0.15

    def _update_eggs(self, dt: float) -> None:
        pl = self._player
        for egg in list(self._eggs):
            egg['vy'] = max(egg['vy'] - _GRAV * dt, -_MAX_VY)
            egg['y'] += egg['vy'] * dt
            landed = False
            for px, py, hw in self._plat_world():
                if abs(self._wrapdx(px, egg['x'])) < hw and py + 0.02 < egg['y'] < py + 0.12:
                    egg['y'] = py + 0.08
                    egg['vy'] = 0.0
                    landed = True
                    break
            if not landed and egg['y'] < _LAVA_Y + 0.06:
                egg['y'] = _LAVA_Y + 0.06
                egg['vy'] = 0.0
            # player collects the egg by flying over it
            if abs(self._wrapdx(egg['x'], pl.x)) < 0.11 and abs(egg['y'] - pl.y) < 0.11:
                self._flash = max(self._flash, 0.4)
                self._eggs.remove(egg)
                continue
            egg['hatch'] -= dt
            if egg['hatch'] <= 0.0:
                self._eggs.remove(egg)
                if len(self._enemies) < _NB - 1:
                    b = _Bird(egg['x'], egg['y'] + 0.06, egg['kind'],
                              1.0 if self.rng.random() < 0.5 else -1.0)
                    self._flap(b)
                    self._enemies.append(b)

    # ---------------------------------------------------------------- update
    def update(self, dt: float, audio: AudioData) -> None:
        """Advance audio envelopes and the whole match simulation."""
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
        self._lava += ((self._bass + self._beat * 0.6) - self._lava) * min(1.0, dt * 8.0)

        tempo = max(0.05, float(self.parameters['speed'])) * (1.0 + self._bass * 0.35)
        self._player_invuln = max(0.0, self._player_invuln - dt)

        self._ai(self._player, dt, tempo)
        self._physics(self._player, dt)
        if self._player.y < _LAVA_Y + 0.04:              # fell in the lava
            self._hit_player()
        for e in list(self._enemies):
            self._ai(e, dt, tempo)
            self._physics(e, dt)
            if e.y < _LAVA_Y + 0.04:
                self._defeat_enemy(e, False)

        self._jousts()
        self._update_eggs(dt)

        act = self._ex[:, 2] >= 0.0
        self._ex[act, 2] += dt * 2.0
        self._ex[self._ex[:, 2] > 1.0] = -1.0

        if not self._enemies and not self._eggs:
            self._wave += 1
            self._flash = max(self._flash, 0.7)
            self._spawn_wave(3 + self._wave)

    # ---------------------------------------------------------------- render
    def render(self) -> None:
        """Upload the match state and draw the neon arena."""
        p = self._prog
        p['iTime'].value = float(self.time)
        p['iResolution'].value = (float(self.width), float(self.height))
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iFlash'].value = self._flash
        p['iHue'].value = self._hue
        p['iGlitter'].value = max(0.0, float(self.parameters['glitter']))
        p['iLavaGlow'].value = float(np.clip(self._lava, 0.0, 2.0))
        p['iPlayerInvuln'].value = 1.0 if self._player_invuln > 0.0 else 0.0

        birds = [self._player] + self._enemies
        pos, face, wing, kind = [], [], [], []
        for i in range(_NB):
            if i < len(birds):
                b = birds[i]
                pos.append((b.x, b.y))
                face.append(b.face)
                wing.append(b.wing)
                kind.append(float(b.kind))
            else:
                pos.append((_OFF, _OFF))
                face.append(1.0)
                wing.append(0.0)
                kind.append(-1.0)
        p['iBirdPos'].value = pos
        p['iBirdFace'].value = face
        p['iBirdWing'].value = wing
        p['iBirdKind'].value = kind

        eggpos, egghatch = [], []
        for i in range(_NEGG):
            if i < len(self._eggs):
                e = self._eggs[i]
                eggpos.append((e['x'], e['y']))
                egghatch.append(float(np.clip(1.0 - e['hatch'] / 5.0, 0.0, 1.0)))
            else:
                eggpos.append((_OFF, _OFF))
                egghatch.append(0.0)
        p['iEgg'].value = eggpos
        p['iEggHatch'].value = egghatch

        plats = list(self._plat_world())
        platval = []
        for i in range(_NPLAT):
            if i < len(plats):
                px, py, hw = plats[i]
                platval.append((px, py, hw, 1.0))
            else:
                platval.append((_OFF, _OFF, 0.1, 0.0))
        p['iPlat'].value = platval
        p['iExpl'].value = [tuple(map(float, e)) for e in self._ex]

        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()

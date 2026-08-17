"""Missile Command — a real trajectory-defense simulation rendered as a neon
light-show.

Same architecture as the rest of the pack: an actual Missile Command match
runs on the CPU (real incoming-missile trajectories, three auto-firing
defense batteries, expanding interceptor blasts that chain-destroy threats,
cities that light up or crater) and the shader renders the true state as
glowing neon. A simple targeting AI has each battery lock the nearest
unintercepted threat and fire when it has ammo, so the sky is always alive
with real, legible missile-command play — waves escalate forever, and a
wiped-out city line quietly rebuilds into the next wave rather than ending
the show.

The backdrop cycles through four cityscape scenes — classic neon, chrome
spires, circuit ruins, and drip fade — each with its own palette, skyline
silhouette, and building ornamentation. A scene switch is triggered whenever
the whole city/battery line is wiped and rebuilt, and also periodically on a
randomised timer, so the skyline keeps evolving through a long session. A
continuous two-layer procedural skyline spans the full screen width behind
the six named cities so the ground is never sparse between them.

Audio reactivity:
  - bass    -> missile speed, blast-radius punch
  - mid     -> battery fire-rate, palette drift
  - treble  -> ember/spark sparkle on blasts and trails
  - beat    -> screen flash, palette shift, salvo burst

Randomised per activation: palette hue, cityscape scene style, wave phase,
battery ammo timers, and ``self.time`` — no two runs look or play alike from
frame 0.
"""
from __future__ import annotations

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

_NCITY = 6
_NBAT = 3
_NENEMY = 14
_NINTER = 9
_NEXPL = 10
_NUFO = 3
_NSTYLE = 4  # 0 classic neon, 1 chrome spires, 2 circuit ruins, 3 drip fade
_OFF = 1000.0

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
uniform float iGroundY;
uniform int   iStyle;
uniform vec2  iCityPos[6];
uniform float iCityAlive[6];
uniform vec2  iBatPos[3];
uniform float iBatAlive[3];
uniform vec2  iEnemyFrom[14];
uniform vec2  iEnemyTo[14];
uniform float iEnemyProg[14];   // <0 inactive
uniform vec2  iInterFrom[9];
uniform vec2  iInterTo[9];
uniform float iInterProg[9];    // <0 inactive
uniform vec3  iExpl[10];        // (x, y, age); age<0 inactive, radius derived from age
uniform vec4  iUfo[3];          // (x, y, alive, driftPhase)
in  vec2 v_uv;
out vec4 fragColor;

#define PI  3.14159265
#define TAU 6.28318530

float hash21(vec2 p) {
    // Bound the input before hashing. Callers add a time term such as
    // floor(t * 20.0); with self.time starting anywhere up to 10000 that
    // reaches ~200000, and the multiply below (x123.34) then lands past
    // float32's usable mantissa -- the field collapses from noise into a
    // fixed lattice, which renders as a regular grid of dots.
    p = mod(p, 512.0);
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

vec3 backdrop(vec2 uv, float A, float t) {
    vec2 p = (uv - 0.5) * vec2(A, 1.0);
    float v = sin(p.x * 2.0 + t * 0.3) + sin(p.y * 1.6 - t * 0.24);
    vec3 col = palette(v * 0.07 + iHue + 0.55 + t * (0.012 + iMid * 0.01));
    col *= col;
    col *= 0.05 + 0.04 * (v * 0.5 + 0.5);
    float sh = hash21(floor(uv * vec2(A, 1.0) * 100.0));
    if (sh > 0.972) {
        float tw = 0.5 + 0.5 * sin(t * (2.0 + sh * 6.0) + sh * 40.0);
        col += vec3(0.6, 0.7, 0.9) * tw * (0.3 + iTreble * 0.4);
    }
    return col;
}

// Continuous procedural skyline spanning the full screen width so the ground
// is never empty between/around the six named cities. Two parallax layers;
// iStyle re-tints and re-decorates it per scene.
vec3 skyline(vec2 p, float t, float groundWorldY) {
    vec3 col = vec3(0.0);
    for (int layer = 0; layer < 2; layer++) {
        float fl = float(layer);
        float freq = 16.0 - fl * 5.0;
        float drift = t * (0.012 + fl * 0.01) * (1.0 + float(iStyle == 3) * 1.6);
        float px = p.x * freq - drift;
        float id = floor(px);
        float fx = fract(px);
        float h = hash21(vec2(id, fl * 3.7 + float(iStyle) * 9.1));
        // Deliberately shorter than the six defended cities below: this is a
        // distant backdrop, and if it out-tops them a levelled city stays
        // hidden behind it and the hit reads as though nothing happened.
        float bh = groundWorldY + (0.016 + h * (0.042 - fl * 0.012));
        float edge = smoothstep(0.0, 0.01, fx) * smoothstep(1.0, 0.99, fx);
        float mask = edge * step(groundWorldY, p.y) * step(p.y, bh);
        vec3 hueCol = palette(iHue + h * 0.18 + fl * 0.08 + float(iStyle) * 0.05);
        col = mix(col, hueCol * (0.20 + fl * 0.30), mask);
        float winH = hash21(vec2(id * 4.0 + floor(fx * 5.0), floor((bh - p.y) * 30.0) + fl * 11.0));
        col += vec3(1.0, 0.8, 0.5) * step(0.72, winH) * mask * (0.12 + 0.12 * sin(t * 3.0 + winH * 30.0));
        if (iStyle == 2) {
            float trace = step(0.985, hash21(vec2(id, floor(p.y * 60.0) + fl))) * mask;
            col += vec3(0.2, 1.0, 0.85) * trace * 1.4;
        }
    }
    return col;
}

float segDist(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

vec3 drawTrail(vec2 p, vec2 from, vec2 to, float prog, vec3 col) {
    if (prog < 0.0) return vec3(0.0);
    vec2 head = mix(from, to, clamp(prog, 0.0, 1.0));
    float trailFrac = clamp(prog, 0.0, 1.0);
    float d = segDist(p, from, head);
    float line = smoothstep(0.010, 0.0, d) * (0.3 + 0.7 * trailFrac);
    float headGlow = exp(-dot(p - head, p - head) * 900.0);
    return col * (line * 0.7 + headGlow * 1.6);
}

// Drip streaks: vertical neon paint bleeding down from a roofline.
float dripStreaks(vec2 q, float bh, float seed, float t) {
    float lane = floor(q.x * 60.0);
    float h = hash21(vec2(lane, seed));
    float len = 0.01 + h * 0.05;
    float flow = fract(t * (0.15 + h * 0.3) + h);
    float dripY = bh - flow * len;
    float d = abs(q.x - (lane + 0.5) / 60.0);
    return smoothstep(0.006, 0.0, d) * smoothstep(dripY - len, dripY, q.y) * step(q.y, bh);
}

// A city block. When destroyed the towers actually come down: each building
// collapses to a jagged, chewed-off stub, the ground under it is cratered and
// scorched, embers wink in the wreckage and smoke drifts up. Recolouring an
// intact skyline (what this used to do) read as "nothing happened" — the
// silhouette has to change for a hit to land visually.
vec3 drawCity(vec2 p, vec2 pos, bool alive, vec3 col, float seed, float tb) {
    vec2 q = p - pos;
    if (abs(q.x) > 0.075 || q.y < -0.020 || q.y > 0.100) return vec3(0.0);
    float bld = 0.0;
    float topY = 0.0;
    for (int i = -3; i <= 3; i++) {
        float bx = float(i) * 0.019;
        float bhFull = 0.032 + 0.046 * hash21(vec2(seed + float(i), 3.0));
        float bw = 0.008 + 0.004 * hash21(vec2(seed + float(i), 7.0));
        float bh = bhFull;
        if (!alive) {
            // Collapsed stub with a torn-off top edge.
            float stub = 0.004 + 0.011 * hash21(vec2(seed + float(i), 11.0));
            float jag = 0.6 + 0.8 * hash21(vec2(floor(q.x * 190.0), seed + 17.0));
            bh = stub * jag;
            bw *= 0.85 + 0.3 * hash21(vec2(seed + float(i), 23.0));
        }
        float m = step(abs(q.x - bx), bw) * step(q.y, bh) * step(-0.002, q.y);
        bld = max(bld, m);
        topY = max(topY, bh * m);
    }

    if (!alive) {
        vec3 c = vec3(0.14, 0.06, 0.05) * bld;
        // Scorched crater dished into the ground line.
        float crater = smoothstep(0.062, 0.0, abs(q.x)) * smoothstep(0.002, -0.016, q.y);
        c += vec3(0.30, 0.09, 0.03) * crater;
        // Embers winking through the rubble.
        float ember = hash21(floor(vec2(q.x * 110.0, tb * 3.0)));
        c += vec3(1.0, 0.38, 0.07) * step(0.955, ember) * bld * 1.6;
        // Smoke drifting off the wreck.
        float sm = hash21(floor(vec2(q.x * 45.0 + tb * 0.6, (q.y - tb * 0.09) * 45.0)));
        c += vec3(0.20, 0.18, 0.20) * step(0.72, sm)
           * smoothstep(0.085, 0.010, q.y) * step(0.008, q.y) * 0.7;
        return c;
    }

    float pulse = 0.8 + 0.4 * sin(iTime * 2.0 + pos.x * 10.0);
    vec3 c = col * pulse * bld;
    float winRow = floor(q.y * 90.0);
    float winCol = floor((q.x + 0.06) * 140.0);
    float winH = hash21(vec2(winCol, winRow + seed * 5.0));
    float win = step(0.55, winH) * bld * step(0.0, q.y);
    c += vec3(1.0, 0.85, 0.5) * win * (0.4 + 0.3 * sin(iTime * 2.5 + winH * 30.0));
    if (iStyle == 1) {
        float spire = smoothstep(0.004, 0.0, abs(q.x)) * step(topY, q.y) * step(q.y, topY + 0.02);
        c += vec3(0.8, 0.95, 1.0) * spire * 1.5;
    }
    if (iStyle == 1 || iStyle == 3) {
        c += col * dripStreaks(q + vec2(0.06, 0.0), topY, seed, iTime) * 1.3;
    }
    return c;
}

// Incoming saucer: hull, glass dome, and a ring of blinking lights.
vec3 drawUfo(vec2 p, vec4 u, float tb) {
    if (u.z < 0.5) return vec3(0.0);
    vec2 q = p - u.xy;
    if (abs(q.x) > 0.06 || abs(q.y) > 0.035) return vec3(0.0);
    vec3 c = vec3(0.0);
    float hull = smoothstep(0.006, -0.004, length(q / vec2(0.045, 0.013)) - 1.0);
    c = mix(c, vec3(0.55, 0.62, 0.75), hull);
    float dome = smoothstep(0.005, -0.003, length((q - vec2(0.0, 0.010)) / vec2(0.019, 0.016)) - 1.0)
               * step(0.008, q.y);
    c = mix(c, vec3(0.35, 1.0, 0.85), dome);
    c += vec3(0.2, 0.9, 0.7) * dome * 0.6;
    // Rim lights chase around the hull.
    for (int i = 0; i < 5; i++) {
        float fi = float(i);
        float lx = (fi - 2.0) * 0.018;
        float blink = 0.5 + 0.5 * sin(tb * 7.0 + fi * 1.6 + u.w);
        c += vec3(1.0, 0.35, 0.5) * blink
           * smoothstep(0.006, 0.0, length(q - vec2(lx, -0.006)));
    }
    // Abduction glow underneath.
    c += vec3(0.3, 1.0, 0.8) * hull * 0.25;
    c += vec3(0.25, 0.9, 0.7)
       * smoothstep(0.030, 0.0, length(q - vec2(0.0, -0.016))) * 0.5;
    return c;
}

vec3 drawBattery(vec2 p, vec2 pos, bool alive, vec3 col) {
    vec2 q = p - pos;
    float body = step(abs(q.x), 0.022) * step(0.0, q.y) * step(q.y, 0.028);
    float dome = smoothstep(0.020, 0.0, length(q - vec2(0.0, 0.028)));
    vec3 c = alive ? col : vec3(0.10, 0.04, 0.04);
    return c * clamp(body + dome, 0.0, 1.0);
}

vec3 drawBlast(vec2 p, vec3 e, vec3 col) {
    if (e.z < 0.0) return vec3(0.0);
    float age = e.z;
    float r = 0.02 + age * (0.16 + iBass * 0.05);
    float d = length(p - e.xy);
    float ring = smoothstep(0.014, 0.0, abs(d - r)) * (1.0 - age);
    float core = exp(-d * d * 90.0) * (1.0 - age * age);
    float spark = pow(0.5 + 0.5 * sin(atan(p.y - e.y, p.x - e.x) * 10.0 + age * 20.0), 6.0)
                * smoothstep(r * 1.4, r * 0.3, d) * (1.0 - age);
    return col * (ring * 1.4 + core * 2.0 + spark * 0.7);
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    vec2 uv = v_uv;
    vec2 p = (uv - 0.5) * vec2(A, 1.0);
    float t = iTime;
    float tb = mod(t, 600.0);   // bounded time for floor()/hash() use
    float groundWorldY = iGroundY - 0.5;

    vec3 col = backdrop(uv, A, t);
    col += skyline(p, t, groundWorldY);

    // Ground line.
    float groundD = uv.y - iGroundY;
    col += mix(vec3(0.15, 0.7, 1.0), palette(iHue), 0.4) * smoothstep(0.004, 0.0, abs(groundD)) * 0.8;
    col = mix(col, vec3(0.02, 0.015, 0.02), smoothstep(0.0, -0.05, groundD));

    for (int i = 0; i < 6; i++) {
        col += drawCity(p, iCityPos[i], iCityAlive[i] > 0.5,
                        palette(iHue + float(i) * 0.06 + 0.1), float(i) * 13.0 + 1.0, tb);
    }
    for (int i = 0; i < 3; i++) {
        col += drawUfo(p, iUfo[i], tb);
    }
    for (int i = 0; i < 3; i++) {
        col += drawBattery(p, iBatPos[i], iBatAlive[i] > 0.5, mix(vec3(0.3, 1.0, 0.6), palette(iHue + 0.4), 0.4));
    }

    vec3 enemyCol = mix(vec3(1.0, 0.25, 0.2), palette(iHue + 0.5), 0.35);
    for (int i = 0; i < 14; i++) {
        col += drawTrail(p, iEnemyFrom[i], iEnemyTo[i], iEnemyProg[i], enemyCol);
    }
    vec3 interCol = mix(vec3(0.25, 0.9, 1.0), palette(iHue + 0.2), 0.35);
    for (int i = 0; i < 9; i++) {
        col += drawTrail(p, iInterFrom[i], iInterTo[i], iInterProg[i], interCol);
    }
    for (int i = 0; i < 10; i++) {
        col += drawBlast(p, iExpl[i], mix(vec3(1.0, 0.75, 0.3), palette(iHue + 0.15), 0.3));
    }

    col += palette(iHue + 0.5) * iFlash * 0.2;
    float sp = hash21(floor(p * iResolution.y * 0.5) + floor(t * 20.0));
    col += vec3(0.9, 0.85, 1.0) * step(0.995, sp) * (0.3 + iTreble * 0.7) * iGlitter;
    // Sharp instant beat pulse, distinct from iFlash's slower explosion decay.
    col += palette(iHue + 0.1) * iBeat * 0.08;

    float vig = smoothstep(1.4, 0.15, length(uv - 0.5));
    col *= vig;
    col = col / (1.0 + col * 0.5);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.85));
    fragColor = vec4(col, 1.0);
}
"""


class MissileCommand(BaseEffect):
    """Self-playing neon Missile Command: a real defense simulation on the CPU."""

    NAME = 'Missile Command'
    AUTHOR = 'unicorn-viz'
    TAGS = ['games', 'arcade', 'neon', 'intense', 'hard', 'energetic']
    PING_PONG_FRIENDS = [
        'Breakout', 'Neon Pac', 'Galaga', 'Joust', 'Tetris',
        'Reactor Breach', 'Audio Spectrum', 'Audio Waveforms',
    ]

    _GROUND_Y = 0.10
    # Cityscape style shown by the most recent instance, across activations in
    # this process. A plain uniform draw on every activation has a 1-in-4
    # chance of repeating the previous style back-to-back, which reads as "it
    # didn't change" even though the pick was genuinely random; excluding the
    # last-shown style from the draw removes that case entirely.
    _last_style: int | None = None

    def _init(self) -> None:
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
        style_choices = [s for s in range(_NSTYLE) if s != MissileCommand._last_style]
        self._style = int(self.rng.choice(style_choices))
        MissileCommand._last_style = self._style
        self._style_timer = float(self.rng.uniform(10.0, 18.0))

        self._city_x = [0.08 + 0.84 * i / (_NCITY - 1) for i in range(_NCITY)]
        self._city_alive = [True] * _NCITY
        self._bat_x = [0.02, 0.5, 0.98]
        self._bat_alive = [True] * _NBAT
        self._bat_cd = [0.0] * _NBAT

        self._enemy = []      # list of dict: from,to,prog,speed
        self._inter = []      # list of dict: from,to,prog,speed,target_enemy
        self._expl = np.full((_NEXPL, 3), -1.0, dtype=np.float32)

        self._ufos: list[dict] = []
        self._ufo_cd = float(self.rng.uniform(3.0, 8.0))

        self._wave = 0
        self._wave_timer = float(self.rng.uniform(0.5, 1.5))
        self._spawn_cd = 0.0

    # ------------------------------------------------------------- helpers
    def _alive_enemies(self):
        """Every undetonated missile, including ones still queued to launch."""
        return [e for e in self._enemy if e['prog'] < 1.0 and not e.get('dead')]

    def _visible_enemies(self):
        """Only missiles actually on screen.

        Wave members spawn with a negative ``prog`` so they stagger in, and
        ``drawTrail`` skips those — but ``_current_pos`` clamps prog to 0, so
        they still report a position up at their launch point. Targeting off
        this list is what made the batteries fire at nothing: they were leading
        missiles that had not appeared yet.
        """
        return [e for e in self._alive_enemies() if e['prog'] >= 0.0]

    def _spawn_wave(self) -> None:
        self._wave += 1
        n = min(3 + self._wave, 9)
        self._enemy = []
        for _ in range(n):
            fx = float(self.rng.uniform(0.05, 0.95))
            tx = float(self.rng.choice(self._live_targets()))
            self._enemy.append({
                'from': (fx, 0.98), 'to': (tx, self._GROUND_Y),
                'prog': float(self.rng.uniform(-1.5, 0.0)),
                'speed': float(self.rng.uniform(0.10, 0.16)),
            })
        self._hue = float((self._hue + self.rng.uniform(0.08, 0.2)) % 1.0)
        self._flash = max(self._flash, 0.4)

    def _choose_style(self) -> None:
        """Switch to a different cityscape scene style."""
        choices = [s for s in range(_NSTYLE) if s != self._style]
        self._style = int(self.rng.choice(choices))
        MissileCommand._last_style = self._style
        self._style_timer = float(self.rng.uniform(10.0, 18.0))

    def _spawn_explosion(self, x: float, y: float) -> None:
        free = np.where(self._expl[:, 2] < 0.0)[0]
        slot = int(free[0]) if free.size else int(np.argmax(self._expl[:, 2]))
        self._expl[slot] = (x, y, 0.0)

    def _live_targets(self) -> list[float]:
        """Ground x positions still worth attacking."""
        t = [self._city_x[i] for i in range(_NCITY) if self._city_alive[i]]
        t += [self._bat_x[i] for i in range(_NBAT) if self._bat_alive[i]]
        return t or list(self._city_x)

    def _update_ufos(self, dt: float, tempo: float) -> None:
        """Drift saucers across the sky; they bomb the ground line as they go."""
        self._ufo_cd -= dt
        if self._ufo_cd <= 0.0 and len(self._ufos) < _NUFO:
            from_left = bool(self.rng.random() < 0.5)
            speed = float(self.rng.uniform(0.07, 0.13))
            self._ufos.append({
                'x': -0.10 if from_left else 1.10,
                'y': float(self.rng.uniform(0.55, 0.86)),
                'vx': speed if from_left else -speed,
                'bomb_cd': float(self.rng.uniform(0.7, 1.8)),
                'phase': float(self.rng.uniform(0.0, 6.28)),
            })
            self._ufo_cd = float(self.rng.uniform(6.0, 14.0))

        for u in list(self._ufos):
            u['x'] += u['vx'] * dt * tempo
            if u['x'] < -0.18 or u['x'] > 1.18:
                self._ufos.remove(u)
                continue
            u['bomb_cd'] -= dt
            if u['bomb_cd'] <= 0.0 and len(self._enemy) < _NENEMY:
                tx = float(self.rng.choice(self._live_targets()))
                self._enemy.append({
                    'from': (float(u['x']), float(u['y'])),
                    'to': (tx, self._GROUND_Y),
                    'prog': 0.0,
                    'speed': float(self.rng.uniform(0.16, 0.26)),
                })
                u['bomb_cd'] = float(self.rng.uniform(1.6, 3.2))

    def _fire_interceptor(self, battery_idx: int, target: tuple[float, float]) -> None:
        if len(self._inter) >= _NINTER:
            return
        origin = (self._bat_x[battery_idx], 0.03)
        self._inter.append({
            'from': origin, 'to': target, 'prog': 0.0,
            'speed': float(self.rng.uniform(0.55, 0.75)),
        })

    # ---------------------------------------------------------------- update
    def update(self, dt: float, audio: AudioData) -> None:
        """Advance audio envelopes and the whole defense simulation."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 2.0))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 2.0))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 2.0))
        if audio.beat > 0.5:
            self._beat = 1.0
            self._flash = max(self._flash, 0.25)
        self._beat = max(0.0, self._beat - dt * 2.5)
        self._flash = max(0.0, self._flash - dt * 1.8)

        tempo = max(0.05, float(self.parameters['speed'])) * (1.0 + self._bass * 0.5)

        # wave spawning
        if not self._alive_enemies() and not self._inter:
            self._wave_timer -= dt
            if self._wave_timer <= 0.0:
                if not any(self._city_alive) and not any(self._bat_alive):
                    self._city_alive = [True] * _NCITY
                    self._bat_alive = [True] * _NBAT
                    self._flash = max(self._flash, 0.7)
                    self._choose_style()
                self._spawn_wave()
                self._wave_timer = float(self.rng.uniform(2.0, 4.0))

        self._style_timer -= dt
        if self._style_timer <= 0.0:
            self._choose_style()

        # advance enemy missiles
        for e in self._enemy:
            if e['prog'] >= 1.0 or e.get('dead'):
                continue
            e['prog'] += dt * e['speed'] * tempo
            if e['prog'] >= 1.0:
                e['prog'] = 1.0
                self._spawn_explosion(*e['to'])
                self._impact(e['to'])

        # battery AI: target nearest live threat, fire on cooldown
        for bi in range(_NBAT):
            self._bat_cd[bi] = max(0.0, self._bat_cd[bi] - dt)
            if not self._bat_alive[bi] or self._bat_cd[bi] > 0.0:
                continue
            # Only shoot at things that are actually on screen — missiles that
            # have launched, and saucers (led a little along their drift).
            aim: list[tuple[float, float]] = [self._predict(e)
                                              for e in self._visible_enemies()]
            aim += [(float(u['x'] + u['vx'] * 0.4), float(u['y'])) for u in self._ufos]
            if not aim:
                continue
            bx = self._bat_x[bi]
            target = min(aim, key=lambda pt: abs(pt[0] - bx))
            self._fire_interceptor(bi, target)
            self._bat_cd[bi] = float(self.rng.uniform(0.5, 1.1)) / (1.0 + self._mid * 0.6)

        # advance interceptors, detonate on arrival
        for m in list(self._inter):
            m['prog'] += dt * m['speed'] * tempo
            if m['prog'] >= 1.0:
                self._spawn_explosion(*m['to'])
                self._inter.remove(m)

        # blast collision: any enemy head inside a fresh blast is destroyed
        for i in range(_NEXPL):
            if self._expl[i, 2] < 0.0 or self._expl[i, 2] > 0.6:
                continue
            ex, ey = float(self._expl[i, 0]), float(self._expl[i, 1])
            radius = 0.02 + float(self._expl[i, 2]) * 0.16
            for e in self._visible_enemies():
                hx, hy = self._current_pos(e)
                if (hx - ex) ** 2 + (hy - ey) ** 2 < radius * radius:
                    e['dead'] = True
                    self._spawn_explosion(hx, hy)
                    self._flash = max(self._flash, 0.15)
            for u in list(self._ufos):
                if (u['x'] - ex) ** 2 + (u['y'] - ey) ** 2 < radius * radius:
                    self._spawn_explosion(float(u['x']), float(u['y']))
                    self._ufos.remove(u)
                    self._flash = max(self._flash, 0.45)

        # age explosions
        act = self._expl[:, 2] >= 0.0
        self._expl[act, 2] += dt * 2.2
        self._expl[self._expl[:, 2] > 1.0] = -1.0

        self._update_ufos(dt, tempo)
        self._spawn_cd = max(0.0, self._spawn_cd - dt)

    def _current_x(self, e: dict) -> float:
        return self._current_pos(e)[0]

    def _current_pos(self, e: dict) -> tuple[float, float]:
        prog = max(0.0, min(1.0, e['prog']))
        fx, fy = e['from']
        tx, ty = e['to']
        return (fx + (tx - fx) * prog, fy + (ty - fy) * prog)

    def _predict(self, e: dict) -> tuple[float, float]:
        """Lead the target a little ahead along its trajectory."""
        prog = max(0.0, min(1.0, e['prog'] + 0.12))
        fx, fy = e['from']
        tx, ty = e['to']
        return (fx + (tx - fx) * prog, fy + (ty - fy) * prog)

    def _impact(self, pos: tuple[float, float]) -> None:
        x = pos[0]
        for i, cx in enumerate(self._city_x):
            if self._city_alive[i] and abs(cx - x) < 0.04:
                self._city_alive[i] = False
                self._flash = max(self._flash, 0.5)
                return
        for i, bx in enumerate(self._bat_x):
            if self._bat_alive[i] and abs(bx - x) < 0.04:
                self._bat_alive[i] = False
                self._flash = max(self._flash, 0.5)
                return

    # ---------------------------------------------------------------- render
    def _wpt(self, x: float, y: float, aspect: float) -> tuple[float, float]:
        """Convert a [0,1]-fraction sim coordinate to the shader's centred,
        aspect-scaled world space (matches ``p = (uv - 0.5) * vec2(A, 1.0)``)."""
        return ((x - 0.5) * aspect, y - 0.5)

    def render(self) -> None:
        """Upload the battle state and draw the neon skyline."""
        p = self._prog
        aspect = float(self.width) / max(float(self.height), 1.0)
        p['iTime'].value = float(self.time)
        p['iResolution'].value = (float(self.width), float(self.height))
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iFlash'].value = self._flash
        p['iHue'].value = self._hue
        p['iGlitter'].value = max(0.0, float(self.parameters['glitter']))
        p['iGroundY'].value = self._GROUND_Y
        p['iStyle'].value = self._style

        p['iCityPos'].value = [self._wpt(x, self._GROUND_Y, aspect) for x in self._city_x]
        p['iCityAlive'].value = [1.0 if a else 0.0 for a in self._city_alive]
        p['iBatPos'].value = [self._wpt(x, self._GROUND_Y, aspect) for x in self._bat_x]
        p['iBatAlive'].value = [1.0 if a else 0.0 for a in self._bat_alive]

        efrom, eto, eprog = [], [], []
        for i in range(_NENEMY):
            if i < len(self._enemy) and self._enemy[i]['prog'] < 1.0 and not self._enemy[i].get('dead'):
                e = self._enemy[i]
                efrom.append(self._wpt(*e['from'], aspect))
                eto.append(self._wpt(*e['to'], aspect))
                eprog.append(float(e['prog']))
            else:
                efrom.append((_OFF, _OFF)); eto.append((_OFF, _OFF)); eprog.append(-1.0)
        p['iEnemyFrom'].value = efrom
        p['iEnemyTo'].value = eto
        p['iEnemyProg'].value = eprog

        ifrom, ito, iprog = [], [], []
        for i in range(_NINTER):
            if i < len(self._inter):
                m = self._inter[i]
                ifrom.append(self._wpt(*m['from'], aspect))
                ito.append(self._wpt(*m['to'], aspect))
                iprog.append(float(m['prog']))
            else:
                ifrom.append((_OFF, _OFF)); ito.append((_OFF, _OFF)); iprog.append(-1.0)
        p['iInterFrom'].value = ifrom
        p['iInterTo'].value = ito
        p['iInterProg'].value = iprog

        p['iExpl'].value = [
            (*self._wpt(float(e[0]), float(e[1]), aspect), float(e[2])) if e[2] >= 0.0
            else (_OFF, _OFF, -1.0)
            for e in self._expl
        ]

        ufo_val = []
        for i in range(_NUFO):
            if i < len(self._ufos):
                u = self._ufos[i]
                ux, uy = self._wpt(float(u['x']), float(u['y']), aspect)
                ufo_val.append((ux, uy, 1.0, float(u['phase'])))
            else:
                ufo_val.append((_OFF, _OFF, 0.0, 0.0))
        p['iUfo'].value = ufo_val

        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()

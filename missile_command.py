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

Audio reactivity:
  - bass    -> missile speed, blast-radius punch
  - mid     -> battery fire-rate, palette drift
  - treble  -> ember/spark sparkle on blasts and trails
  - beat    -> screen flash, palette shift, salvo burst

Randomised per activation: palette hue, wave phase, battery ammo timers, and
``self.time`` — no two runs look or play alike from frame 0.
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

vec3 drawCity(vec2 p, vec2 pos, bool alive, vec3 col) {
    vec2 q = p - pos;
    if (abs(q.x) > 0.05 || q.y < -0.005 || q.y > 0.045) return vec3(0.0);
    float bld = 0.0;
    for (int i = -2; i <= 2; i++) {
        float bx = float(i) * 0.018;
        float bh = 0.02 + 0.012 * hash21(vec2(pos.x * 50.0 + float(i), 3.0));
        float m = step(abs(q.x - bx), 0.007) * step(q.y, bh);
        bld = max(bld, m);
    }
    vec3 c = alive ? col * (0.8 + 0.4 * sin(iTime * 2.0 + pos.x * 10.0)) : vec3(0.12, 0.05, 0.04);
    return c * bld;
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
    vec2 p = vec2(uv.x * A, uv.y);
    float t = iTime;

    vec3 col = backdrop(uv, A, t);

    // Ground line.
    float groundD = uv.y - iGroundY;
    col += mix(vec3(0.15, 0.7, 1.0), palette(iHue), 0.4) * smoothstep(0.004, 0.0, abs(groundD)) * 0.8;
    col = mix(col, vec3(0.02, 0.015, 0.02), smoothstep(0.0, -0.05, groundD));

    for (int i = 0; i < 6; i++) {
        col += drawCity(p, iCityPos[i], iCityAlive[i] > 0.5, palette(iHue + float(i) * 0.06 + 0.1));
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

        self._city_x = [0.20 + 0.6 * i / (_NCITY - 1) for i in range(_NCITY)]
        self._city_alive = [True] * _NCITY
        self._bat_x = [0.06, 0.5, 0.94]
        self._bat_alive = [True] * _NBAT
        self._bat_cd = [0.0] * _NBAT

        self._enemy = []      # list of dict: from,to,prog,speed
        self._inter = []      # list of dict: from,to,prog,speed,target_enemy
        self._expl = np.full((_NEXPL, 3), -1.0, dtype=np.float32)

        self._wave = 0
        self._wave_timer = float(self.rng.uniform(0.5, 1.5))
        self._spawn_cd = 0.0

    # ------------------------------------------------------------- helpers
    def _alive_enemies(self):
        return [e for e in self._enemy if e['prog'] < 1.0 and not e.get('dead')]

    def _spawn_wave(self) -> None:
        self._wave += 1
        n = min(3 + self._wave, 9)
        self._enemy = []
        for _ in range(n):
            fx = float(self.rng.uniform(0.05, 0.95))
            targets = [self._city_x[i] for i in range(_NCITY) if self._city_alive[i]]
            targets += [self._bat_x[i] for i in range(_NBAT) if self._bat_alive[i]]
            if not targets:
                targets = list(self._city_x)
            tx = float(self.rng.choice(targets))
            self._enemy.append({
                'from': (fx, 0.98), 'to': (tx, self._GROUND_Y),
                'prog': float(self.rng.uniform(-1.5, 0.0)),
                'speed': float(self.rng.uniform(0.10, 0.16)),
            })
        self._hue = float((self._hue + self.rng.uniform(0.08, 0.2)) % 1.0)
        self._flash = max(self._flash, 0.4)

    def _spawn_explosion(self, x: float, y: float) -> None:
        free = np.where(self._expl[:, 2] < 0.0)[0]
        slot = int(free[0]) if free.size else int(np.argmax(self._expl[:, 2]))
        self._expl[slot] = (x, y, 0.0)

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
                self._spawn_wave()
                self._wave_timer = float(self.rng.uniform(2.0, 4.0))

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
            threats = self._alive_enemies()
            if not threats:
                continue
            bx = self._bat_x[bi]
            nearest = min(threats, key=lambda e: abs(self._current_x(e) - bx))
            predicted = self._predict(nearest)
            self._fire_interceptor(bi, predicted)
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
            for e in self._alive_enemies():
                hx, hy = self._current_pos(e)
                if (hx - ex) ** 2 + (hy - ey) ** 2 < radius * radius:
                    e['dead'] = True
                    self._spawn_explosion(hx, hy)
                    self._flash = max(self._flash, 0.15)

        # age explosions
        act = self._expl[:, 2] >= 0.0
        self._expl[act, 2] += dt * 2.2
        self._expl[self._expl[:, 2] > 1.0] = -1.0

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
    def render(self) -> None:
        """Upload the battle state and draw the neon skyline."""
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
        p['iGroundY'].value = self._GROUND_Y

        p['iCityPos'].value = [(x, self._GROUND_Y) for x in self._city_x]
        p['iCityAlive'].value = [1.0 if a else 0.0 for a in self._city_alive]
        p['iBatPos'].value = [(x, self._GROUND_Y) for x in self._bat_x]
        p['iBatAlive'].value = [1.0 if a else 0.0 for a in self._bat_alive]

        efrom, eto, eprog = [], [], []
        for i in range(_NENEMY):
            if i < len(self._enemy) and self._enemy[i]['prog'] < 1.0 and not self._enemy[i].get('dead'):
                e = self._enemy[i]
                efrom.append(e['from']); eto.append(e['to']); eprog.append(float(e['prog']))
            else:
                efrom.append((_OFF, _OFF)); eto.append((_OFF, _OFF)); eprog.append(-1.0)
        p['iEnemyFrom'].value = efrom
        p['iEnemyTo'].value = eto
        p['iEnemyProg'].value = eprog

        ifrom, ito, iprog = [], [], []
        for i in range(_NINTER):
            if i < len(self._inter):
                m = self._inter[i]
                ifrom.append(m['from']); ito.append(m['to']); iprog.append(float(m['prog']))
            else:
                ifrom.append((_OFF, _OFF)); ito.append((_OFF, _OFF)); iprog.append(-1.0)
        p['iInterFrom'].value = ifrom
        p['iInterTo'].value = ito
        p['iInterProg'].value = iprog

        p['iExpl'].value = [tuple(map(float, e)) for e in self._expl]

        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()

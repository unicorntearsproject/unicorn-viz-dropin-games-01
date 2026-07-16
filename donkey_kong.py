"""Donkey Kong — a real girder-climbing platformer simulation rendered as a
neon light-show.

Same architecture as the rest of the pack: an actual climb runs on the CPU
(five zig-zag girder platforms, ladders, a self-playing hero who paths
toward the next ladder up and hops over rolling barrels, real barrel physics
rolling downhill and dropping a level at each girder's low end) and the
shader renders the true state as glowing neon. Kong sits up top, hurling a
fresh barrel down the girders on a timer; when the hero reaches the top the
whole rig flashes in celebration and resets for another climb — perpetual,
so the show never actually "ends."

Audio reactivity:
  - bass    -> barrel roll speed, girder pulse
  - mid     -> hero climb speed, palette drift
  - treble  -> spark/rivet sparkle on girders and barrels
  - beat    -> screen flash, palette shift, Kong throw emphasis

Randomised per activation: palette hue, hero starting phase, barrel-throw
timer, and ``self.time`` — no two runs look or play alike from frame 0.
"""
from __future__ import annotations

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

_NROWS = 5
_NBARREL = 8
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
uniform vec4  iRow[5];        // (x0, y0, x1, y1) girder endpoints, world space
uniform vec4  iLadder[8];     // (x, yBottom, yTop, _) unused w
uniform int   iLadderCount;
uniform vec2  iHeroPos;
uniform float iHeroFace;
uniform float iHeroClimb;     // 0 walking, 1 climbing
uniform float iHeroHit;       // death-flash blink
uniform vec2  iKongPos;
uniform float iKongThrow;     // 0..1 throw-arm swing envelope
uniform vec2  iBarrelPos[8];
uniform float iBarrelRot[8];
uniform float iBarrelAlive[8];
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
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

vec3 backdrop(vec2 uv, float A, float t) {
    vec2 p = (uv - 0.5) * vec2(A, 1.0);
    float v = sin(p.x * 2.1 + t * 0.3) + sin(p.y * 1.8 - t * 0.24);
    vec3 col = palette(v * 0.08 + iHue + 0.6 + t * 0.014);
    col *= col;
    col *= 0.05 + 0.05 * (v * 0.5 + 0.5);
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

vec3 drawGirder(vec2 p, vec4 row, vec3 col) {
    vec2 a = row.xy, b = row.zw;
    float d = segDist(p, a, b);
    float body = smoothstep(0.018, 0.010, d);
    float rivetPhase = fract(dot(p - a, normalize(b - a)) * 14.0);
    float rivet = smoothstep(0.06, 0.0, min(rivetPhase, 1.0 - rivetPhase)) * body;
    vec3 o = col * body * 0.7;
    o += col * rivet * 0.6;
    o += col * smoothstep(0.020, 0.0, d) * 0.4;
    return o;
}

vec3 drawLadder(vec2 p, vec4 lad, vec3 col) {
    float x = lad.x, y0 = lad.y, y1 = lad.z;
    if (p.y < y0 - 0.01 || p.y > y1 + 0.01) return vec3(0.0);
    float rail = smoothstep(0.007, 0.0, abs(abs(p.x - x) - 0.018));
    float rung = smoothstep(0.006, 0.0, abs(fract((p.y - y0) * 9.0) - 0.5) - 0.35)
               * step(abs(p.x - x), 0.02);
    return col * clamp(rail + rung, 0.0, 1.0) * 0.8;
}

// Small glowing hero sprite, local space q in roughly [-0.3,0.3]^2, y up.
vec3 drawHero(vec2 q, float face, float climb, float t) {
    vec3 col = vec3(0.0);
    vec3 body = mix(vec3(1.0, 0.25, 0.35), vec3(1.0, 0.8, 0.3), 0.4);
    float torso = smoothstep(0.05, -0.01, length((q - vec2(0.0, 0.02)) / vec2(0.09, 0.13)) - 1.0);
    float head = smoothstep(0.05, -0.01, length(q - vec2(0.0, 0.17)) - 0.075);
    col = mix(col, body, torso);
    col = mix(col, vec3(1.0, 0.85, 0.65), head);
    float legPhase = sin(t * (climb > 0.5 ? 14.0 : 10.0)) * 0.06;
    vec2 lq = q - vec2(0.0, -0.14);
    float legs = smoothstep(0.03, -0.01, segDist(lq, vec2(-0.02, 0.0), vec2(-0.02 + legPhase, -0.10)))
               + smoothstep(0.03, -0.01, segDist(lq, vec2(0.02, 0.0), vec2(0.02 - legPhase, -0.10)));
    col += body * clamp(legs, 0.0, 1.0);
    float armSwing = climb > 0.5 ? abs(sin(t * 14.0)) * 0.08 : 0.02;
    vec2 aq = q - vec2(face * 0.07, 0.03);
    float arm = smoothstep(0.025, -0.01, segDist(aq, vec2(0.0, 0.0), vec2(face * 0.05, -0.08 - armSwing)));
    col += body * arm;
    col += (torso + head) * 0.3;
    return col;
}

vec3 drawKong(vec2 q, float throwT) {
    vec3 col = vec3(0.0);
    vec3 fur = vec3(0.55, 0.28, 0.12);
    float body = smoothstep(0.06, -0.02, length((q - vec2(0.0, 0.0)) / vec2(0.20, 0.16)) - 1.0);
    float head = smoothstep(0.05, -0.02, length(q - vec2(0.0, 0.20)) - 0.11);
    col = mix(col, fur, body);
    col = mix(col, fur * 1.15, head);
    col = mix(col, vec3(0.85, 0.65, 0.45), smoothstep(0.05, -0.01, length(q - vec2(0.0, 0.18)) - 0.06));
    float armLift = sin(throwT * PI);
    vec2 aq = rot(armLift * 1.2) * (q - vec2(0.16, 0.10));
    float arm = smoothstep(0.03, -0.01, segDist(aq, vec2(0.0, 0.0), vec2(0.0, 0.16)));
    col += fur * arm;
    return col;
}

vec3 drawBarrel(vec2 q, float spin, vec3 col) {
    float body = smoothstep(0.03, -0.01, length(q / vec2(0.045, 0.038)) - 1.0);
    float band = smoothstep(0.010, 0.0, abs(abs(fract(q.x * 12.0 + spin) - 0.5) - 0.5)) * body;
    vec3 o = col * body * 0.6;
    o += col * (1.0 - band) * body * 0.5;
    o += col * smoothstep(0.04, -0.005, length(q / vec2(0.045, 0.038)) - 1.0) * 0.3;
    return o;
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    vec2 uv = v_uv;
    vec2 p = (uv - 0.5) * vec2(A, 1.0);
    float t = iTime;

    vec3 col = backdrop(uv, A, t);

    for (int i = 0; i < 5; i++) {
        col += drawGirder(p, iRow[i], mix(vec3(0.2, 0.6, 1.0), palette(iHue + float(i) * 0.06), 0.5));
    }
    for (int i = 0; i < iLadderCount; i++) {
        col += drawLadder(p, iLadder[i], mix(vec3(1.0, 0.85, 0.3), palette(iHue + 0.3), 0.35));
    }

    for (int i = 0; i < 8; i++) {
        if (iBarrelAlive[i] < 0.5) continue;
        vec2 bq = p - iBarrelPos[i];
        col += drawBarrel(bq, iBarrelRot[i], mix(vec3(0.85, 0.5, 0.2), palette(iHue + 0.5), 0.3))
             * (1.0 + iBass * 0.3);
    }

    vec2 hq = rot(0.0) * (p - iHeroPos);
    hq.x *= iHeroFace >= 0.0 ? 1.0 : -1.0;
    float heroVis = (iHeroHit > 0.5) ? step(0.5, fract(t * 14.0)) : 1.0;
    col += drawHero(hq, sign(iHeroFace + 0.001), iHeroClimb, t) * heroVis
         * (1.0 + iMid * 0.2);

    vec2 kq = p - iKongPos;
    col += drawKong(kq, iKongThrow) * (1.0 + iBeat * 0.25);

    col += palette(iHue + 0.5) * iFlash * 0.22;
    float sp = hash21(floor(p * iResolution.y * 0.5) + floor(t * 20.0));
    col += vec3(0.9, 0.85, 1.0) * step(0.995, sp) * (0.3 + iTreble * 0.7) * iGlitter;

    float vig = smoothstep(1.4, 0.15, length(uv - 0.5));
    col *= vig;
    col = col / (1.0 + col * 0.5);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.85));
    fragColor = vec4(col, 1.0);
}
"""


class DonkeyKong(BaseEffect):
    """Self-playing neon Donkey Kong: a real girder-climb simulation."""

    NAME = 'Donkey Kong'
    AUTHOR = 'unicorn-viz'
    TAGS = ['games', 'arcade', 'neon', 'energetic', 'groovy', 'hard']
    PING_PONG_FRIENDS = [
        'Breakout', 'Neon Pac', 'Galaga', 'Joust', 'Tetris', 'Missile Command',
        'Audio Spectrum', 'Audio Waveforms',
    ]

    _HERO_SPEED = 0.55         # progress-fraction per second along a girder
    _CLIMB_SPEED = 0.9
    _BARREL_SPEED = 0.32

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

        # Five zig-zag girders: row 0 (bottom/start) .. row 4 (top/Kong).
        # World space here is the shader's centred p-coordinate
        # (p = (uv - 0.5) * aspect), NOT [0,1] UV space — x=0 is screen
        # centre, so the rig must span roughly [-0.55, 0.55] to stay
        # visible across typical aspect ratios (an earlier version used
        # [0.1, 0.9] verbatim as if it were UV space, pushing the whole
        # rig off-screen to the right).
        left, right = -0.55, 0.55
        span = right - left
        self._rows = []
        for i in range(_NROWS):
            y = -0.42 + i * 0.21
            if i % 2 == 0:
                x0, y0, x1, y1 = left, y, right, y + 0.07
            else:
                x0, y0, x1, y1 = left, y + 0.07, right, y
            self._rows.append((x0, y0, x1, y1))

        # Ladders: two per adjacent-row gap, at fixed fractional positions.
        self._ladders = []          # (row_lo, t_lo_frac_x, row_hi)
        fracs = [0.30, 0.70]
        for i in range(_NROWS - 1):
            for j, fr in enumerate(fracs):
                x = left + span * (fr if (i + j) % 2 == 0 else 1.0 - fr)
                self._ladders.append({'row_lo': i, 'row_hi': i + 1, 'x': x})

        self._hero_row = 0
        self._hero_t = float(self.rng.uniform(0.05, 0.15))
        self._hero_climb = None       # active ladder dict while climbing
        self._hero_climb_frac = 0.0
        self._hero_face = 1.0
        self._hero_jump = 0.0
        self._hero_hit = 0.0

        self._kong_pos = self._row_point(_NROWS - 1, 0.08)
        self._kong_throw = 0.0
        self._throw_cd = float(self.rng.uniform(1.0, 2.5))

        self._barrels: list[dict] = []
        self._win_flash = 0.0

    # ------------------------------------------------------------- geometry
    def _row_point(self, row: int, tfrac: float) -> tuple[float, float]:
        x0, y0, x1, y1 = self._rows[row]
        tfrac = float(np.clip(tfrac, 0.0, 1.0))
        return (x0 + (x1 - x0) * tfrac, y0 + (y1 - y0) * tfrac)

    def _row_downhill_dir(self, row: int) -> float:
        """+1 if the row descends toward t=1 (low end on the right), else -1."""
        x0, y0, x1, y1 = self._rows[row]
        return 1.0 if y1 < y0 else -1.0

    # ------------------------------------------------------------------ AI
    def _nearest_up_ladder(self, row: int, t: float):
        cands = [ladder for ladder in self._ladders if ladder['row_lo'] == row]
        if not cands:
            return None
        x0, y0, x1, y1 = self._rows[row]
        cur_x = x0 + (x1 - x0) * t
        return min(cands, key=lambda ladder: abs(ladder['x'] - cur_x))

    def _t_at_x(self, row: int, x: float) -> float:
        x0, _, x1, _ = self._rows[row]
        return float(np.clip((x - x0) / max(x1 - x0, 1e-6), 0.0, 1.0))

    def _hero_pos(self) -> tuple[float, float]:
        if self._hero_climb is not None:
            lad = self._hero_climb
            rl, rh = lad['row_lo'], lad['row_hi']
            y0 = self._row_point(rl, self._t_at_x(rl, lad['x']))[1]
            y1 = self._row_point(rh, self._t_at_x(rh, lad['x']))[1]
            y = y0 + (y1 - y0) * self._hero_climb_frac
        else:
            x, y = self._row_point(self._hero_row, self._hero_t)
            jump_off = 0.10 * np.sin(np.pi * self._hero_jump) if self._hero_jump > 0.0 else 0.0
            return (x, y + jump_off)
        return (lad['x'], y)

    def _barrel_near(self, row: int, t: float, ahead_sign: float) -> bool:
        for b in self._barrels:
            if b['row'] != row:
                continue
            dt = (b['t'] - t) * ahead_sign
            if 0.0 < dt < 0.09:
                return True
        return False

    def _update_hero(self, dt: float, tempo: float) -> None:
        self._hero_jump = max(0.0, self._hero_jump - dt * 2.2)

        if self._hero_climb is not None:
            self._hero_climb_frac += dt * self._CLIMB_SPEED * tempo
            if self._hero_climb_frac >= 1.0:
                self._hero_row = self._hero_climb['row_hi']
                self._hero_t = self._t_at_x(self._hero_row, self._hero_climb['x'])
                self._hero_climb = None
                if self._hero_row == _NROWS - 1:
                    self._win_flash = 1.0
                    self._flash = max(self._flash, 0.8)
                    self._hue = float((self._hue + self.rng.uniform(0.1, 0.25)) % 1.0)
                    self._hero_row = 0
                    self._hero_t = float(self.rng.uniform(0.05, 0.15))
            return

        target = self._nearest_up_ladder(self._hero_row, self._hero_t)
        if target is None:
            return
        target_t = self._t_at_x(self._hero_row, target['x'])
        ahead = 1.0 if target_t > self._hero_t else -1.0
        self._hero_face = ahead
        if self._barrel_near(self._hero_row, self._hero_t, ahead) and self._hero_jump <= 0.0:
            self._hero_jump = 1.0
        step = ahead * self._HERO_SPEED * tempo * dt
        if abs(target_t - self._hero_t) < abs(step):
            self._hero_t = target_t
            self._hero_climb = target
            self._hero_climb_frac = 0.0
        else:
            self._hero_t += step

        # Barrel collision (only while grounded, not mid-jump).
        if self._hero_hit <= 0.0 and self._hero_jump <= 0.0:
            for b in self._barrels:
                if b['row'] == self._hero_row and abs(b['t'] - self._hero_t) < 0.025:
                    self._hero_hit = 1.0
                    self._flash = max(self._flash, 0.5)
                    self._hero_row = 0
                    self._hero_t = float(self.rng.uniform(0.05, 0.15))
                    self._hero_climb = None
                    break

    def _update_barrels(self, dt: float, tempo: float) -> None:
        self._throw_cd -= dt
        if self._throw_cd <= 0.0 and len(self._barrels) < _NBARREL:
            row = _NROWS - 1
            t0 = 0.06 if self._row_downhill_dir(row) > 0 else 0.94
            self._barrels.append({'row': row, 't': t0, 'rot': 0.0})
            self._kong_throw = 1.0
            self._throw_cd = float(self.rng.uniform(1.6, 3.0)) / (1.0 + self._bass * 0.5)

        for b in list(self._barrels):
            d = self._row_downhill_dir(b['row'])
            b['t'] += d * dt * self._BARREL_SPEED * tempo
            b['rot'] += dt * self._BARREL_SPEED * tempo * 6.0
            if b['t'] < 0.0 or b['t'] > 1.0:
                if b['row'] == 0:
                    self._barrels.remove(b)
                else:
                    b['row'] -= 1
                    b['t'] = float(np.clip(b['t'], 0.0, 1.0))

    # ---------------------------------------------------------------- update
    def update(self, dt: float, audio: AudioData) -> None:
        """Advance audio envelopes and the whole climb simulation."""
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
        self._hero_hit = max(0.0, self._hero_hit - dt * 1.4)
        self._kong_throw = max(0.0, self._kong_throw - dt * 1.6)

        tempo = max(0.05, float(self.parameters['speed'])) * (1.0 + self._mid * 0.4)
        self._update_hero(dt, tempo)
        self._update_barrels(dt, tempo * (1.0 + self._bass * 0.3))

    # ---------------------------------------------------------------- render
    def render(self) -> None:
        """Upload the climb state and draw the neon rig."""
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
        p['iRow'].value = [tuple(map(float, r)) for r in self._rows]

        ladder_val = [(ladder['x'],
                      self._row_point(ladder['row_lo'], self._t_at_x(ladder['row_lo'], ladder['x']))[1],
                      self._row_point(ladder['row_hi'], self._t_at_x(ladder['row_hi'], ladder['x']))[1],
                      0.0)
                     for ladder in self._ladders]
        while len(ladder_val) < 8:
            ladder_val.append((_OFF, _OFF, _OFF, 0.0))
        p['iLadder'].value = ladder_val
        p['iLadderCount'].value = len(self._ladders)

        hx, hy = self._hero_pos()
        p['iHeroPos'].value = (hx, hy)
        p['iHeroFace'].value = float(self._hero_face)
        p['iHeroClimb'].value = 1.0 if self._hero_climb is not None else 0.0
        p['iHeroHit'].value = 1.0 if self._hero_hit > 0.0 else 0.0

        p['iKongPos'].value = self._kong_pos
        p['iKongThrow'].value = float(self._kong_throw)

        bpos, brot, balive = [], [], []
        for i in range(_NBARREL):
            if i < len(self._barrels):
                b = self._barrels[i]
                bpos.append(self._row_point(b['row'], b['t']))
                brot.append(float(b['rot']))
                balive.append(1.0)
            else:
                bpos.append((_OFF, _OFF))
                brot.append(0.0)
                balive.append(0.0)
        p['iBarrelPos'].value = bpos
        p['iBarrelRot'].value = brot
        p['iBarrelAlive'].value = balive

        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()

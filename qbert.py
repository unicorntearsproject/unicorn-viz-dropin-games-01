"""Q*bert — a real isometric pyramid-hopping simulation rendered as a neon
light-show.

Same architecture as the rest of the pack: an actual climb runs on the CPU
(a real 7-row/28-cube isometric pyramid, diagonal hopping, per-cube colour
state that changes on landing) and the shader renders the true state as
glowing neon cubes. A self-playing hero hops diagonally toward the nearest
not-yet-recoloured cube; a chasing enemy spawns partway through each level
and hops toward the hero on its own timer. A cube-to-cube collision costs
the hero a life-flash and resets him to the apex; clearing every cube to
its target colour flashes the whole pyramid and starts the next level —
perpetual, so the show never actually "ends."

Audio reactivity:
  - bass    -> hop speed, pyramid pulse
  - mid     -> enemy hop rate, palette drift
  - treble  -> cube-face sparkle
  - beat    -> screen flash, palette shift, landing emphasis

Randomised per activation: palette hue, hero starting phase, enemy spawn
timer, and ``self.time`` — no two runs look or play alike from frame 0.
"""
from __future__ import annotations

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

_ROWS = 7
_NCUBE = 28          # 1+2+3+4+5+6+7
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
uniform vec2  iCubePos[28];
uniform float iCubeState[28];   // 0 start colour, 1 target colour
uniform vec2  iHeroPos;
uniform float iHeroHop;         // 0..1 hop-arc phase
uniform float iHeroHit;
uniform vec2  iEnemyPos;
uniform float iEnemyHop;
uniform float iEnemyAlive;
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
    float v = sin(p.x * 2.2 + t * 0.3) + sin(p.y * 1.8 - t * 0.24);
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

// Isometric cube: diamond top face + two darker side faces.
vec3 drawCube(vec2 q, vec3 topCol, float pulse) {
    float hw = 0.058, hh = 0.034, side = 0.030;
    vec3 o = vec3(0.0);

    // Top diamond face.
    vec2 tq = q - vec2(0.0, side * 0.5);
    float diamond = abs(tq.x) / hw + abs(tq.y) / hh;
    float top = step(diamond, 1.0);
    vec3 topShade = topCol * (0.85 + pulse * 0.35);
    o = mix(o, topShade, top);

    // Left face (darker).
    vec2 lq = q - vec2(-hw * 0.5, -side * 0.5 - hh * 0.5);
    float leftFace = step(abs(lq.x) / (hw * 0.5) + 0.001, 1.0) * step(abs(lq.y + side * 0.0), side + hh * 0.5)
                   * step(q.x, 0.0) * step(-side - hh, q.y) * step(q.y, side * 0.0);
    // Simplify with a trapezoid-ish box instead for robustness:
    float lf = step(-hw, q.x) * step(q.x, 0.0) * step(-hh - side, q.y) * step(q.y, -hh + (q.x + hw) / hw * hh);
    o = mix(o, topCol * (0.35 + pulse * 0.2), lf);

    float rf = step(0.0, q.x) * step(q.x, hw) * step(-hh - side, q.y) * step(q.y, -hh + (hw - q.x) / hw * hh);
    o = mix(o, topCol * (0.5 + pulse * 0.25), rf);

    // Edge glow on the top diamond outline.
    float edge = smoothstep(0.06, 0.0, abs(diamond - 1.0)) * top;
    o += topCol * edge * 0.6;
    return o;
}

vec3 drawHero(vec2 q, float hop) {
    vec3 col = vec3(1.0, 0.55, 0.15);
    float bounce = sin(hop * PI) * 0.04;
    vec2 hq = q - vec2(0.0, 0.03 + bounce);
    float body = smoothstep(0.03, -0.005, length(hq / vec2(0.032, 0.038)) - 1.0);
    vec3 o = col * body;
    o += vec3(1.0) * smoothstep(0.010, 0.0, length(hq - vec2(-0.010, 0.010))) * body * 0.7;
    o += vec3(1.0) * smoothstep(0.010, 0.0, length(hq - vec2(0.010, 0.010))) * body * 0.7;
    o += col * body * 0.4;
    return o;
}

vec3 drawEnemy(vec2 q, float hop, vec3 col) {
    float bounce = sin(hop * PI) * 0.04;
    vec2 eq = q - vec2(0.0, 0.03 + bounce);
    float body = smoothstep(0.028, -0.005, length(eq / vec2(0.030, 0.034)) - 1.0);
    vec3 o = col * body * 0.7;
    o += col * body * body * 0.5;
    return o;
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    vec2 uv = v_uv;
    vec2 p = (uv - 0.5) * vec2(A, 1.0);
    float t = iTime;

    vec3 col = backdrop(uv, A, t);

    for (int i = 0; i < 28; i++) {
        vec2 cq = p - iCubePos[i];
        if (abs(cq.x) > 0.09 || cq.y < -0.12 || cq.y > 0.08) continue;
        vec3 startCol = mix(vec3(0.15, 0.10, 0.35), palette(iHue + 0.55), 0.5);
        vec3 targetCol = mix(vec3(1.0, 0.75, 0.15), palette(iHue), 0.5);
        vec3 cubeCol = mix(startCol, targetCol, iCubeState[i]);
        float pulse = 0.5 + 0.5 * sin(t * 2.0 + float(i) * 0.4);
        col += drawCube(cq, cubeCol, pulse * 0.3 + iBass * 0.3 + iCubeState[i] * 0.2);
    }

    vec2 hq = p - iHeroPos;
    float heroVis = (iHeroHit > 0.5) ? step(0.5, fract(t * 14.0)) : 1.0;
    col += drawHero(hq, iHeroHop) * heroVis * (1.0 + iMid * 0.2);

    if (iEnemyAlive > 0.5) {
        vec2 eq = p - iEnemyPos;
        col += drawEnemy(eq, iEnemyHop, mix(vec3(0.6, 0.2, 1.0), palette(iHue + 0.5), 0.35))
             * (1.0 + iBeat * 0.3);
    }

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


class QBert(BaseEffect):
    """Self-playing neon Q*bert: a real isometric pyramid-hop simulation."""

    NAME = 'Q*bert'
    AUTHOR = 'unicorn-viz'
    TAGS = ['games', 'arcade', 'neon', 'energetic', 'groovy', 'intense']
    PING_PONG_FRIENDS = [
        'Breakout', 'Neon Pac', 'Galaga', 'Joust', 'Tetris',
        'Missile Command', 'Donkey Kong', 'Audio Spectrum',
    ]

    _HOP_TIME = 0.28

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

        dx, dy, top_y = 0.13, 0.11, 0.42
        self._cubes: list[tuple[int, int]] = []
        self._cube_pos: dict[tuple[int, int], tuple[float, float]] = {}
        for row in range(_ROWS):
            for c in range(row + 1):
                self._cubes.append((row, c))
                x = (c - row / 2.0) * dx
                y = top_y - row * dy
                self._cube_pos[(row, c)] = (x, y)
        self._cube_state = {rc: 0.0 for rc in self._cubes}

        self._hero_cell = (0, 0)
        self._hero_from = self._cube_pos[self._hero_cell]
        self._hero_to = self._hero_from
        self._hero_hop_t = 1.0
        self._hero_hit = 0.0

        self._enemy_cell: tuple[int, int] | None = None
        self._enemy_from = (0.0, 0.0)
        self._enemy_to = (0.0, 0.0)
        self._enemy_hop_t = 1.0
        self._enemy_cd = float(self.rng.uniform(3.0, 6.0))
        self._enemy_move_cd = 0.0

        self._level_flash = 0.0

    # ------------------------------------------------------------------ AI
    def _neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        row, c = cell
        cands = [(row + 1, c), (row + 1, c + 1), (row - 1, c - 1), (row - 1, c)]
        return [(r, cc) for r, cc in cands if 0 <= r < _ROWS and 0 <= cc <= r]

    def _hero_pos(self) -> tuple[float, float]:
        t = min(1.0, self._hero_hop_t)
        fx, fy = self._hero_from
        tx, ty = self._hero_to
        x = fx + (tx - fx) * t
        y = fy + (ty - fy) * t + 0.05 * np.sin(np.pi * t)
        return (x, y)

    def _enemy_pos(self) -> tuple[float, float]:
        t = min(1.0, self._enemy_hop_t)
        fx, fy = self._enemy_from
        tx, ty = self._enemy_to
        x = fx + (tx - fx) * t
        y = fy + (ty - fy) * t + 0.05 * np.sin(np.pi * t)
        return (x, y)

    def _start_hero_hop(self, dest: tuple[int, int]) -> None:
        self._hero_from = self._cube_pos[self._hero_cell]
        self._hero_to = self._cube_pos[dest]
        self._hero_cell = dest
        self._hero_hop_t = 0.0

    def _update_hero(self, dt: float, tempo: float) -> None:
        if self._hero_hop_t < 1.0:
            self._hero_hop_t += dt / self._HOP_TIME * tempo
            if self._hero_hop_t >= 1.0:
                self._cube_state[self._hero_cell] = 1.0
                self._flash = max(self._flash, 0.15)
            return
        uncleared = [rc for rc, v in self._cube_state.items() if v < 0.5]
        if not uncleared:
            self._level_flash = 1.0
            self._flash = max(self._flash, 0.8)
            self._hue = float((self._hue + self.rng.uniform(0.1, 0.25)) % 1.0)
            self._cube_state = {rc: 0.0 for rc in self._cubes}
            self._hero_cell = (0, 0)
            self._hero_from = self._hero_to = self._cube_pos[self._hero_cell]
            self._enemy_cell = None
            self._enemy_cd = float(self.rng.uniform(3.0, 6.0))
            return
        opts = self._neighbors(self._hero_cell)
        if not opts:
            return
        target = min(uncleared, key=lambda rc: (rc[0] - self._hero_cell[0]) ** 2
                     + (rc[1] - self._hero_cell[1]) ** 2)
        best = min(opts, key=lambda o: (o[0] - target[0]) ** 2 + (o[1] - target[1]) ** 2)
        self._start_hero_hop(best)

    def _update_enemy(self, dt: float, tempo: float) -> None:
        if self._enemy_cell is None:
            self._enemy_cd -= dt
            if self._enemy_cd <= 0.0:
                self._enemy_cell = (0, 0)
                self._enemy_from = self._enemy_to = self._cube_pos[self._enemy_cell]
                self._enemy_hop_t = 1.0
                self._enemy_move_cd = 0.0
            return
        if self._enemy_hop_t < 1.0:
            self._enemy_hop_t += dt / (self._HOP_TIME * 1.3) * tempo
            return
        self._enemy_move_cd -= dt
        if self._enemy_move_cd <= 0.0:
            opts = self._neighbors(self._enemy_cell)
            if opts:
                best = min(opts, key=lambda o: (o[0] - self._hero_cell[0]) ** 2
                          + (o[1] - self._hero_cell[1]) ** 2)
                self._enemy_from = self._cube_pos[self._enemy_cell]
                self._enemy_to = self._cube_pos[best]
                self._enemy_cell = best
                self._enemy_hop_t = 0.0
            self._enemy_move_cd = float(self.rng.uniform(0.4, 0.8)) / (1.0 + self._mid * 0.5)

        if self._enemy_cell == self._hero_cell and self._hero_hit <= 0.0:
            self._hero_hit = 1.0
            self._flash = max(self._flash, 0.5)
            self._hero_cell = (0, 0)
            self._hero_from = self._hero_to = self._cube_pos[self._hero_cell]
            self._hero_hop_t = 1.0
            self._enemy_cell = None
            self._enemy_cd = float(self.rng.uniform(3.0, 6.0))

    # ---------------------------------------------------------------- update
    def update(self, dt: float, audio: AudioData) -> None:
        """Advance audio envelopes and the whole pyramid simulation."""
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
        self._level_flash = max(0.0, self._level_flash - dt * 1.5)

        tempo = max(0.05, float(self.parameters['speed'])) * (1.0 + self._bass * 0.5)
        self._update_hero(dt, tempo)
        self._update_enemy(dt, tempo)

    # ---------------------------------------------------------------- render
    def render(self) -> None:
        """Upload the pyramid state and draw the neon cubes."""
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

        p['iCubePos'].value = [self._cube_pos[rc] for rc in self._cubes]
        p['iCubeState'].value = [float(self._cube_state[rc]) for rc in self._cubes]

        p['iHeroPos'].value = self._hero_pos()
        p['iHeroHop'].value = float(min(1.0, self._hero_hop_t))
        p['iHeroHit'].value = 1.0 if self._hero_hit > 0.0 else 0.0

        if self._enemy_cell is not None:
            p['iEnemyPos'].value = self._enemy_pos()
            p['iEnemyHop'].value = float(min(1.0, self._enemy_hop_t))
            p['iEnemyAlive'].value = 1.0
        else:
            p['iEnemyPos'].value = (_OFF, _OFF)
            p['iEnemyHop'].value = 0.0
            p['iEnemyAlive'].value = 0.0

        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()

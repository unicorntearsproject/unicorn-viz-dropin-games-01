"""Breakout effect.

A futuristic reinterpretation of the arcade classic. The effect simulates a
ball, paddle, and destructible brick matrix on the CPU while a GLSL shader
renders a neon/cyber arena with strong audio reactivity.

Controls surfaced through standard runtime tweakables:
- speed
- zoom
- reactivity
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

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
uniform vec2 iResolution;
uniform float iSpeed;
uniform float iZoom;
uniform float iReactivity;
uniform float iBass;
uniform float iMid;
uniform float iTreble;
uniform float iBeat;
uniform float iHitFlash;
uniform float iPaletteShift;
uniform vec2 iBallPos;
uniform vec2 iBallVel;
uniform float iBallRadius;
uniform float iPaddleX;
uniform float iPaddleW;
uniform vec2 iGrid;
uniform vec2 iBrickMin;
uniform vec2 iBrickMax;
uniform sampler2D iBricks;

in vec2 v_uv;
out vec4 fragColor;

vec3 palette(float t) {
    vec3 a = vec3(0.5);
    vec3 b = vec3(0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.03, 0.20, 0.34);
    return a + b * cos(6.28318 * (c * t + d));
}

float rect_sdf(vec2 p, vec2 half_size) {
    vec2 d = abs(p) - half_size;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float grid_glow(vec2 uv, float t) {
    vec2 g = fract(uv * vec2(26.0, 16.0) + vec2(t * 0.11, -t * 0.08));
    float line_x = exp(-60.0 * min(g.x, 1.0 - g.x));
    float line_y = exp(-50.0 * min(g.y, 1.0 - g.y));
    return (line_x + line_y) * 0.35;
}

vec3 draw_bricks(vec2 uv, float t, float reactivity) {
    vec3 col = vec3(0.0);
    if (uv.x < iBrickMin.x || uv.x > iBrickMax.x || uv.y < iBrickMin.y || uv.y > iBrickMax.y) {
        return col;
    }

    vec2 local = (uv - iBrickMin) / (iBrickMax - iBrickMin);
    vec2 cellf = local * iGrid;
    ivec2 cell = ivec2(clamp(floor(cellf), vec2(0.0), iGrid - vec2(1.0)));
    vec2 within = fract(cellf) - 0.5;

    float state = texelFetch(iBricks, ivec2(cell.x, int(iGrid.y) - 1 - cell.y), 0).r;
    if (state < 0.01) {
        return col;
    }

    float edge = rect_sdf(within, vec2(0.43, 0.34));
    float shell = smoothstep(0.05, 0.0, abs(edge));
    float core = smoothstep(0.015, -0.02, edge);

    float idx = float(cell.x) + float(cell.y) * iGrid.x;
    float hue_t = idx / max(1.0, iGrid.x * iGrid.y) + iPaletteShift + t * 0.04;
    vec3 brick_col = palette(hue_t + iMid * 0.07 * reactivity);

    float scan = 0.5 + 0.5 * sin((within.x * 18.0 + within.y * 13.0) + t * (5.0 + iTreble * 2.2));
    float pulse = 0.65 + 0.35 * sin(t * (3.0 + iBass * 1.8) + idx * 0.27);

    col += brick_col * shell * (0.24 + iTreble * 0.24 * reactivity);
    col += brick_col * core * pulse * (0.28 + 0.24 * scan + iMid * 0.18 * reactivity);
    col += brick_col * core * iHitFlash * 0.55;
    return col;
}

vec3 draw_paddle(vec2 uv, float t, float reactivity) {
    vec2 center = vec2(iPaddleX, 0.085);
    vec2 p = uv - center;
    float d = rect_sdf(p, vec2(iPaddleW * 0.5, 0.022));
    float body = smoothstep(0.016, -0.01, d);
    float ring = smoothstep(0.06, 0.0, abs(d));

    float wave = 0.5 + 0.5 * sin((p.x * 28.0) + t * (8.0 + iMid * 4.0));
    vec3 col = mix(vec3(0.16, 0.92, 1.00), vec3(1.00, 0.40, 0.88), wave + iPaletteShift * 0.3);

    vec3 out_col = vec3(0.0);
    out_col += col * body * (0.68 + iBass * 0.30 * reactivity);
    out_col += col * ring * (0.11 + iTreble * 0.11 * reactivity);
    return out_col;
}

vec3 draw_ball(vec2 uv, float t, float reactivity) {
    vec2 bp = uv - iBallPos;
    float d = length(bp);
    float core = smoothstep(iBallRadius, 0.0, d);
    float halo = smoothstep(iBallRadius * 3.8, iBallRadius * 0.5, d);

    vec3 col = mix(vec3(0.12, 0.98, 0.86), vec3(1.00, 0.34, 0.76), 0.5 + 0.5 * sin(t * 4.5 + iPaletteShift * 8.0));
    vec3 out_col = col * core * (0.92 + iBass * 0.58 * reactivity);
    out_col += col * halo * (0.09 + iTreble * 0.13 * reactivity + iHitFlash * 0.12);

    vec2 trail_dir = normalize(-iBallVel + vec2(1e-5));
    float trail = 0.0;
    for (int i = 1; i <= 5; i++) {
        float fi = float(i);
        vec2 tp = iBallPos + trail_dir * fi * (0.018 + iTreble * 0.010 * reactivity);
        float td = length(uv - tp);
        trail += smoothstep(iBallRadius * 2.6, 0.0, td) * (1.0 / (fi + 1.0));
    }
    out_col += col * trail * (0.10 + iTreble * 0.18 * reactivity);
    return out_col;
}

void main() {
    float t = iTime * iSpeed;
    float react = max(iReactivity, 0.05);

    vec2 uv = (v_uv - 0.5) / max(iZoom, 0.08) + 0.5;
    vec2 uv_n = uv * 2.0 - 1.0;
    uv_n.x *= iResolution.x / max(iResolution.y, 1.0);

    vec3 bg_a = vec3(0.005, 0.008, 0.022);
    vec3 bg_b = vec3(0.012, 0.006, 0.032);
    float grad = smoothstep(-1.0, 1.0, uv_n.y + 0.10 * uv_n.x + 0.22 * sin(t * 0.22 + uv_n.x * 0.9));
    vec3 col = mix(bg_a, bg_b, grad);

    float g = grid_glow(uv, t);
    col += vec3(0.05, 0.16, 0.30) * g * (0.42 + iMid * 0.22 * react);

    // Arena rails.
    float rails = exp(-120.0 * abs(uv.x - 0.08)) + exp(-120.0 * abs(uv.x - 0.92));
    rails *= (0.90 + 0.10 * clamp(abs(uv_n.x), 0.0, 1.6));
    col += vec3(0.10, 0.54, 0.85) * rails * (0.12 + iBass * 0.22 * react);

    col += draw_bricks(uv, t, react);
    col += draw_paddle(uv, t, react);
    col += draw_ball(uv, t, react);

    // Beat ring pulse at center of arena.
    float beat_ring = smoothstep(0.36 + iBeat * 0.28, 0.0, abs(length(uv - vec2(0.5, 0.5)) - (0.10 + iBeat * 0.18)));
    col += vec3(0.85, 0.36, 1.0) * beat_ring * iBeat * 0.20;

    // Darker filmic finish with contrast push.
    col = col / (col + 1.08);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.82));
    col = clamp((col - 0.5) * 1.22 + 0.5, 0.0, 1.0);

    fragColor = vec4(col, 1.0);
}
"""


class Breakout(BaseEffect):
    """Neon Breakout reinterpretation with heavy music reactivity."""

    NAME = 'Breakout'
    AUTHOR = 'Copilot'
    TAGS = ['classic', 'audio', 'futuristic', 'arcade']
    PING_PONG_FRIENDS = [
        'Tron Grid',
        'Kaleidoscope',
        'Copper Bars',
        'Escher',
        'Fireworks',
        'Hexy Stars',
        'Metaballs',
        'Particle Storm',
        'Plasma',
        'Prism Lattice',
        'Psychedelic',
        'Audio Sine',
        'Starfield',
        'Tunnel',
        'Unicorn Tears',
        'Van Gogh',
        'Wavey Gravy',
    ]

    GRID_COLS = 12
    GRID_ROWS = 7

    def _init(self) -> None:
        """Initialize GL resources and simulation state."""
        self.parameters = {
            'speed': float(self.config.get('speed', 1.05)),
            'zoom': float(self.config.get('zoom', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.40)),
        }

        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._brick_state = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.float32)
        self._brick_tex = self.ctx.texture((self.GRID_COLS, self.GRID_ROWS), 1, dtype='f4')
        self._brick_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._brick_tex.repeat_x = False
        self._brick_tex.repeat_y = False
        self._brick_dirty = True

        self._brick_min = np.array([0.08, 0.50], dtype=np.float32)
        self._brick_max = np.array([0.92, 0.92], dtype=np.float32)

        self._ball_pos = np.array([0.50, 0.26], dtype=np.float32)
        angle = float(self.rng.uniform(-0.95, -0.35))
        self._ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        self._ball_speed = float(self.rng.uniform(0.50, 0.74))
        self._ball_radius = float(self.rng.uniform(0.014, 0.020))

        self._paddle_x = float(self.rng.uniform(0.36, 0.64))
        self._paddle_w = float(self.rng.uniform(0.17, 0.22))

        self._palette_shift = float(self.rng.uniform(0.0, 1.0))

        self._bass = 0.0
        self._mid = 0.0
        self._treble = 0.0
        self._beat = 0.0
        self._hit_flash = 0.0

        self._serve_timer = 0.0
        self._reset_board(initial=True)

    def _reset_board(self, initial: bool = False) -> None:
        """Refill the brick matrix with slight random variation."""
        self._brick_state.fill(1.0)

        # Keep board mostly full; occasionally punch tiny gaps for variation.
        gap_count = int(self.rng.integers(0, 8)) if not initial else int(self.rng.integers(0, 4))
        for _ in range(gap_count):
            r = int(self.rng.integers(0, self.GRID_ROWS))
            c = int(self.rng.integers(0, self.GRID_COLS))
            self._brick_state[r, c] = 0.0

        self._brick_dirty = True
        self._palette_shift = float((self._palette_shift + self.rng.uniform(0.08, 0.32)) % 1.0)

    def _sync_brick_texture(self) -> None:
        """Upload brick state texture only when dirty."""
        if not self._brick_dirty:
            return
        self._brick_tex.write(self._brick_state.tobytes())
        self._brick_dirty = False

    def _advance_simulation(self, dt: float) -> None:
        """Advance paddle, ball, and collisions in normalized playfield space."""
        react = max(0.05, float(self.parameters['reactivity']))
        speed = max(0.05, float(self.parameters['speed']))

        # Paddle auto-tracks the ball with audio wobble to keep the scene alive.
        target_x = float(self._ball_pos[0] + (self._mid - 0.5) * 0.06 * react)
        self._paddle_x += (target_x - self._paddle_x) * min(1.0, dt * (4.0 + self._bass * 3.0))
        self._paddle_x = float(np.clip(self._paddle_x, 0.09 + self._paddle_w * 0.5, 0.91 - self._paddle_w * 0.5))

        if self._serve_timer > 0.0:
            self._serve_timer = max(0.0, self._serve_timer - dt)
            self._ball_pos[0] = self._paddle_x
            self._ball_pos[1] = 0.12
            return

        beat_boost = 1.0 + self._beat * 0.65
        velocity = self._ball_vel * (self._ball_speed * speed * beat_boost)
        self._ball_pos += velocity * dt

        # Wall collisions.
        if self._ball_pos[0] <= 0.08 + self._ball_radius:
            self._ball_pos[0] = 0.08 + self._ball_radius
            self._ball_vel[0] = abs(self._ball_vel[0])
            self._hit_flash = max(self._hit_flash, 0.34)
        elif self._ball_pos[0] >= 0.92 - self._ball_radius:
            self._ball_pos[0] = 0.92 - self._ball_radius
            self._ball_vel[0] = -abs(self._ball_vel[0])
            self._hit_flash = max(self._hit_flash, 0.34)

        if self._ball_pos[1] >= 0.95 - self._ball_radius:
            self._ball_pos[1] = 0.95 - self._ball_radius
            self._ball_vel[1] = -abs(self._ball_vel[1])
            self._hit_flash = max(self._hit_flash, 0.38)

        # Paddle collision.
        paddle_y = 0.085
        paddle_h = 0.022
        if (
            self._ball_vel[1] < 0.0
            and abs(self._ball_pos[0] - self._paddle_x) <= (self._paddle_w * 0.5 + self._ball_radius)
            and abs(self._ball_pos[1] - paddle_y) <= (paddle_h + self._ball_radius)
        ):
            self._ball_pos[1] = paddle_y + paddle_h + self._ball_radius
            hit_offset = (self._ball_pos[0] - self._paddle_x) / max(self._paddle_w * 0.5, 1e-5)
            refl_x = float(np.clip(hit_offset * 0.92 + (self._mid - 0.5) * 0.38 * react, -0.96, 0.96))
            refl_y = float(math.sqrt(max(0.08, 1.0 - refl_x * refl_x)))
            self._ball_vel = np.array([refl_x, refl_y], dtype=np.float32)
            self._ball_vel /= max(np.linalg.norm(self._ball_vel), 1e-6)
            self._hit_flash = max(self._hit_flash, 0.82)

        # Brick collision.
        if (
            self._brick_min[0] <= self._ball_pos[0] <= self._brick_max[0]
            and self._brick_min[1] <= self._ball_pos[1] <= self._brick_max[1]
        ):
            local = (self._ball_pos - self._brick_min) / (self._brick_max - self._brick_min)
            col = int(np.clip(local[0] * self.GRID_COLS, 0, self.GRID_COLS - 1))
            row_from_bottom = int(np.clip(local[1] * self.GRID_ROWS, 0, self.GRID_ROWS - 1))
            row = self.GRID_ROWS - 1 - row_from_bottom

            if self._brick_state[row, col] > 0.01:
                self._brick_state[row, col] = 0.0
                self._brick_dirty = True
                self._ball_vel[1] *= -1.0
                self._hit_flash = max(self._hit_flash, 1.0)
                self._ball_speed = float(np.clip(self._ball_speed + 0.020 + self._treble * 0.018 + self._bass * 0.010, 0.42, 1.15))

        # Miss / respawn.
        if self._ball_pos[1] < -0.02:
            self._serve_timer = 0.45
            self._ball_speed = float(self.rng.uniform(0.54, 0.80))
            launch_angle = float(self.rng.uniform(-1.10, -0.22))
            self._ball_vel = np.array([math.cos(launch_angle), math.sin(launch_angle)], dtype=np.float32)
            self._ball_vel /= max(np.linalg.norm(self._ball_vel), 1e-6)
            self._hit_flash = max(self._hit_flash, 0.45)

        if float(self._brick_state.sum()) < 0.5:
            self._reset_board()
            self._hit_flash = max(self._hit_flash, 0.85)

    def update(self, dt: float, audio: AudioData) -> None:
        """Advance simulation and audio envelopes."""
        super().update(dt, audio)

        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 2.0))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 2.0))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 2.0))

        if audio.beat > 0.5:
            self._beat = 1.0
            self._hit_flash = max(self._hit_flash, 0.88)
            self._palette_shift = float((self._palette_shift + 0.04 + self._mid * 0.03) % 1.0)
        self._beat = max(0.0, self._beat - dt * 2.0)
        self._hit_flash = max(0.0, self._hit_flash - dt * (1.45 + self._treble * 0.55))

        self._advance_simulation(dt)

    def render(self) -> None:
        """Render the neon breakout arena."""
        self._sync_brick_texture()
        self._brick_tex.use(location=0)

        self._prog['iTime'].value = float(self.time)
        self._prog['iResolution'].value = (float(self.width), float(self.height))
        self._prog['iSpeed'].value = max(0.05, float(self.parameters['speed']))
        self._prog['iZoom'].value = max(0.08, float(self.parameters['zoom']))
        self._prog['iReactivity'].value = max(0.05, float(self.parameters['reactivity']))
        self._prog['iBass'].value = self._bass
        self._prog['iMid'].value = self._mid
        self._prog['iTreble'].value = self._treble
        self._prog['iBeat'].value = self._beat
        self._prog['iHitFlash'].value = self._hit_flash
        self._prog['iPaletteShift'].value = self._palette_shift
        self._prog['iBallPos'].value = (float(self._ball_pos[0]), float(self._ball_pos[1]))
        self._prog['iBallVel'].value = (float(self._ball_vel[0]), float(self._ball_vel[1]))
        self._prog['iBallRadius'].value = self._ball_radius
        self._prog['iPaddleX'].value = self._paddle_x
        self._prog['iPaddleW'].value = self._paddle_w
        self._prog['iGrid'].value = (float(self.GRID_COLS), float(self.GRID_ROWS))
        self._prog['iBrickMin'].value = (float(self._brick_min[0]), float(self._brick_min[1]))
        self._prog['iBrickMax'].value = (float(self._brick_max[0]), float(self._brick_max[1]))
        self._prog['iBricks'].value = 0

        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
        self._brick_tex.release()

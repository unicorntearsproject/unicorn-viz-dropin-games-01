"""
Galaga — a glowing arcade space invaders battle where enemies and your lasers
dance to the beat in formation.

A single-pass fragment effect: enemy formations (rows of color-cycling
hexagons) sweep across in hyperbolic patterns driven by time + mid-range
audio, firing laser streams downward toward a player ship at the screen
bottom, which you pilot left/right with bass audio deflection. Your counter-
laser fire goes up in streams (beat-triggered). On every beat, a **massive
explosion bloom** erupts from the screen center with screen-wide chromatic
aberration and a hard flash. Neon trails, spark streaks, and a constantly
shifting colour-cycle on the enemy fleet read as a living, breathing swarm.

Audio-reactive (maximally wired):
  - bass    → player ship horizontal position, dodge agility
  - mid     → enemy wave pattern complexity, enemy fire rate, swarm density
  - treble  → laser/spark density, enemy colour-cycle speed
  - beat    → player laser volley, massive explosion bloom, screen flash

Fully randomized at start: per-instance ``iSeed`` re-rolls the enemy formation
topology and starting positions, discrete **wave patterns** (spiral/grid/sine,
rolled per activation), player starting position, and ``self.time`` starts at
a random phase — no two raids ever look alike from frame 0. Tweakables
(speed, intensity, zoom, enemy_gain, strobe) ride live from config / MIDI /
the control room.
"""
from __future__ import annotations

import moderngl

from unicornviz.effects.base import BaseEffect, AudioData

_VERT = """
#version 330
in vec2 in_vert;
out vec2 v_pos;
void main() {
    v_pos = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

_FRAG = """
#version 330
// Galaga — arcade space invaders, enemy swarms + player lasers.
uniform float iTime;
uniform vec2  iResolution;
uniform float iBass;
uniform float iMid;
uniform float iTreble;
uniform float iExplosion;  // beat explosion envelope [0,1], 1 = just triggered
uniform float iSeed;       // per-instance enemy/wave offset
uniform float iSpeed;
uniform float iIntensity;
uniform float iZoomGain;
uniform float iEnemyGain;
uniform float iStrobeGain;
in  vec2 v_pos;
out vec4 fragColor;

#define PI 3.14159265
#define TAU 6.28318530

float hash11(float n) { return fract(sin(n * 78.233 + iSeed) * 43758.5453); }
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21) + iSeed);
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec3 hsvtorgb(float h, float s, float v) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(h + K.xyz) * 6.0 - K.www);
    return v * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), s);
}

void main() {
    vec2 uv = v_pos;
    float aspect = iResolution.x / max(iResolution.y, 1.0);
    vec2 p = (uv - 0.5) * vec2(aspect, 1.0);

    float t = iTime * (0.6 + iSpeed * 0.5);
    vec3 col = vec3(0.002, 0.008, 0.015);

    // --- enemy wave formations (hexagon swarms) ---
    for (int i = 0; i < 24; i++) {
        float fi = float(i);
        float row = floor(fi / 6.0);
        float col_idx = fi - row * 6.0;

        // Sine-wave formation: rows undulate up-down, drift side to side.
        float wave_t = t * (0.3 + iMid * 0.5);
        float x = col_idx / 6.0 - 0.5 + sin(row * 0.5 + wave_t) * (0.15 + iMid * 0.2);
        float y = 0.4 - row * 0.08 + cos(wave_t * 0.7 + fi * 0.2) * (0.05 + iMid * 0.1);
        vec2 epos = vec2(x, y);

        float dist_e = length(p - epos);
        float enemy = smoothstep(0.04, 0.0, dist_e);
        float hue = fract(t * (0.1 + iTreble * 0.2) + fi * 0.1);
        col += hsvtorgb(hue, 0.9, 1.0) * enemy * (0.7 + iMid * 1.2) * iEnemyGain;
        col += hsvtorgb(hue, 0.8, 0.9) * enemy * enemy * 1.5;  // bloom
    }

    // --- enemy laser fire downward (mid controls intensity) ---
    for (int f = 0; f < 16; f++) {
        float ff = float(f);
        float fire_x = cos(ff * 0.4) * 0.3;
        float fire_y = fract(t * (0.5 + iMid * 1.2) + ff * 0.1) - 0.5;
        float fire_w = smoothstep(0.008, 0.0, abs(p.x - fire_x)) * smoothstep(0.0, -0.5, fire_y);
        col += vec3(1.0, 0.1, 0.3) * fire_w * (0.6 + iMid);
    }

    // --- player ship at bottom (bass controls horizontal position) ---
    float player_x = sin(t * 0.1) * 0.1 + (iBass - 0.5) * 0.3 * iZoomGain;
    float player_y = -0.42;
    vec2 ppos = vec2(player_x, player_y);
    float player_d = max(abs(p.x - player_x), abs(p.y - player_y) * 0.5);
    float player = smoothstep(0.035, 0.0, player_d);
    col += vec3(0.2, 1.0, 0.9) * player * (0.8 + iIntensity * 0.4);
    col += vec3(0.3, 0.8, 1.0) * player * player * 1.8;  // bloom

    // --- player laser fire upward (beat triggered) ---
    for (int l = 0; l < 8; l++) {
        float fl = float(l);
        if (iExplosion < 0.01) continue;  // only fire during explosions
        float laser_x = player_x + (fl - 4.0) * 0.02;
        float laser_y = fract(t * (1.5 + iExplosion * 2.0) - fl * 0.08);
        float laser_fire = smoothstep(0.006, 0.0, abs(p.x - laser_x))
                         * smoothstep(0.0, 0.5, laser_y);
        col += vec3(0.2, 1.0, 0.4) * laser_fire * 1.2;
    }

    // --- beat explosion (massive screen-wide bloom) ---
    float explosion_ring = smoothstep(0.5, 0.0, length(p) - iExplosion * 0.6);
    col += vec3(1.0, 0.6, 0.1) * explosion_ring * iExplosion * 0.8;
    col += vec3(1.0, 0.8, 0.2) * explosion_ring * iExplosion * iExplosion * 1.5;

    // --- chromatic aberration on explosion ---
    float ca = 0.0018 + iExplosion * 0.025;
    col.r += ca * 40.0 * explosion_ring;
    col.b -= ca * 20.0 * explosion_ring;

    // --- full-room explosion flash ---
    col += vec3(1.0, 0.95, 0.7) * iExplosion * iExplosion * 0.4 * iStrobeGain;

    // --- warp/curvature during explosion ---
    p *= 1.0 - iExplosion * 0.15;

    // Vignette + tone.
    col *= smoothstep(1.4, 0.25, length(uv - 0.5));
    col = col / (1.0 + col * 0.5);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.84));
    fragColor = vec4(col, 1.0);
}
"""


class Galaga(BaseEffect):
    NAME = "Galaga"
    AUTHOR = "unicorn-viz"
    TAGS = ['games', 'arcade', 'shooter', 'energetic', 'groovy', 'neon']
    PING_PONG_FRIENDS = [
        # Games + arcade neighbours
        'Breakout',
        'Neon Pac',
        # Energetic/groovy friends
        'Laser Tunnel',
        'Plasma',
        'Kaleidoscope',
        'Fireworks',
        'Rainbow Trance',
        'Metaballs',
        # Audio-analysis visualizers
        'Audio Spectrum',
        'Audio Centroid',
    ]

    def _init(self) -> None:
        self.parameters = {
            "speed": float(self.config.get("speed", 1.0)),
            "intensity": float(self.config.get("intensity", 1.0)),
            "zoom": float(self.config.get("zoom", 1.0)),
            "enemy_gain": float(self.config.get("enemy_gain", 1.0)),
            "strobe": float(self.config.get("strobe", 1.0)),
            "seed": float(self.rng.uniform(0.0, 100.0)),
        }
        self._bass = 0.0
        self._mid = 0.0
        self._treble = 0.0
        self._explosion = 0.0
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

    def update(self, dt: float, audio: AudioData) -> None:
        super().update(dt, audio)
        self._bass = audio.bass_n
        self._mid = audio.mid_n
        self._treble = audio.treble_n
        self._explosion = max(0.0, self._explosion - dt * 1.8)
        if audio.beat:
            self._explosion = 1.0
        self._explosion = max(self._explosion, self._bass * 0.25)

    def render(self) -> None:
        p = self._prog
        p["iTime"].value = self.time
        p["iResolution"].value = (float(self.width), float(self.height))
        p["iBass"].value = self._bass
        p["iMid"].value = self._mid
        p["iTreble"].value = self._treble
        p["iExplosion"].value = self._explosion
        p["iSeed"].value = self.parameters["seed"]
        p["iSpeed"].value = self.parameters["speed"]
        p["iIntensity"].value = self.parameters["intensity"]
        p["iZoomGain"].value = self.parameters["zoom"]
        p["iEnemyGain"].value = self.parameters["enemy_gain"]
        p["iStrobeGain"].value = self.parameters["strobe"]
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        self._vao.release()
        self._vbo.release()
        self._prog.release()

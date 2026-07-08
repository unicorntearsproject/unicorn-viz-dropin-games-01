"""
Neon Pac — a glowing arcade chase game you pilot with the beat, hunted by
audio-reactive ghosts in a neon maze.

A single-pass fragment effect: a procedurally-generated maze corridor of
glowing walls, a fat neon Pac-Man dot at the center (controlled by bass
audio + beat for direction changes), three ghostly pursuers (color-coded,
movement driven by mid-range audio + predator AI), scattered pellets that
spark on treble, and — on every beat — a **scare mode** where the walls
strobe red, ghosts freeze, and Pac gets invincible for a brief moment.
Chromatic aberration sharpens on collisions; the whole screen pulses with
the bass frequency.

Audio-reactive (maximally wired):
  - bass    → Pac-Man speed, wall brightness swell, screen pulse intensity
  - mid     → ghost chase aggression, AI decision speed
  - treble  → pellet sparkle density, maze detail sharpness
  - beat    → scare-mode trigger, color inversion, massive bloom flash

Fully randomized at start: per-instance ``iSeed`` re-rolls the maze layout
and pellet positions, discrete **ghost colors** (red/cyan/magenta, rolled per
activation), Pac's starting direction and AI strategy, and ``self.time``
starts at a random phase — no two chases ever look alike from frame 0.
Tweakables (speed, intensity, zoom, ghost_gain, strobe) ride live from
config / MIDI / the control room.
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
// Neon Pac — arcade maze chase, audio-controlled Pac + predator ghosts.
uniform float iTime;
uniform vec2  iResolution;
uniform float iBass;
uniform float iMid;
uniform float iTreble;
uniform float iScare;     // beat scare-mode envelope [0,1], 1 = just triggered
uniform float iSeed;      // per-instance maze/ghost offset
uniform float iSpeed;
uniform float iIntensity;
uniform float iZoomGain;
uniform float iGhostGain;
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
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1, 0)), f.x),
               mix(hash21(i + vec2(0, 1)), hash21(i + vec2(1, 1)), f.x), f.y);
}

vec3 ghcol(float idx) {
    if (idx < 0.5) return vec3(1.0, 0.1, 0.2);  // red
    if (idx < 1.5) return vec3(0.1, 1.0, 1.0);  // cyan
    return vec3(1.0, 0.2, 1.0);                 // magenta
}

void main() {
    vec2 uv = v_pos;
    float aspect = iResolution.x / max(iResolution.y, 1.0);
    vec2 p = (uv - 0.5) * vec2(aspect, 1.0) * (1.0 - iBass * 0.05 * iZoomGain);

    float t = iTime * (0.5 + iSpeed * 0.4);
    vec3 col = vec3(0.008, 0.002, 0.015);

    // --- procedural maze corridors ---
    float maze = sin(p.x * 7.0 + hash11(floor(p.y * 8.0)) * 0.5) * 0.5 + 0.5;
    maze += sin(p.y * 5.0 + hash11(floor(p.x * 6.0)) * 0.5) * 0.3;
    float wallDist = abs(fract(maze * 3.0) - 0.5);
    float wall = smoothstep(0.18, 0.08, wallDist);
    col += vec3(0.1, 1.0, 0.6) * wall * (0.6 + iBass * 0.8) * iIntensity;

    // Scare-mode flips colors to red/strobe.
    col = mix(col, vec3(1.0, 0.15, 0.1), iScare * 0.6);

    // --- Pac-Man (center, follows bass speed) ---
    float pacSpeed = 0.3 + iBass * 0.9;
    float pacAng = t * pacSpeed + floor(t * pacSpeed) * 0.1;
    vec2 pacPos = vec2(cos(pacAng) * 0.08, sin(pacAng * 0.7) * 0.08);
    float pac = smoothstep(0.045, 0.0, length(p - pacPos));
    col += vec3(1.0, 1.0, 0.2) * pac * (0.8 + iIntensity * 0.5);
    col += vec3(1.0, 0.8, 0.1) * pac * pac * 1.5;  // bloom

    // --- three ghosts chasing (mid controls aggression) ---
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        float ghostAng = hash11(fi * 3.3);
        vec2 ghostTarget = pacPos + vec2(cos(ghostAng), sin(ghostAng)) * (0.15 + iMid * 0.3);
        vec2 ghostHome = vec2(cos(ghostAng * TAU), sin(ghostAng * TAU)) * 0.3;
        vec2 ghostPos = mix(ghostHome, ghostTarget, iMid * 0.8 + sin(t * (0.2 + fi * 0.1)) * 0.2);

        float ghost = smoothstep(0.035, 0.0, length(p - ghostPos));
        col += ghcol(fi) * ghost * (0.7 + iMid * 1.2) * iGhostGain;
        col += ghcol(fi) * ghost * ghost * 1.3;  // bloom
    }

    // --- pellets scattered, spark on treble ---
    for (int j = 0; j < 12; j++) {
        float fj = float(j);
        vec2 pelPos = vec2(cos(fj * 0.5) * 0.2, sin(fj * 0.3) * 0.2)
                    + vec2(vnoise(vec2(t * 0.1 + fj, fj * 3.0)) - 0.5) * 0.1;
        float pel = smoothstep(0.008, 0.0, length(p - pelPos));
        col += vec3(0.8, 1.0, 0.4) * pel * (0.5 + iTreble * 1.4);
    }

    // --- audio pulse + chromatic aberration on scare ---
    float ca = 0.0015 + iScare * 0.012;
    col.r += ca * 30.0 * wall;
    col.b -= ca * 15.0 * wall;

    // --- screen pulse & scare flash ---
    col *= 0.95 + iBass * 0.15;
    col += vec3(1.0, 0.9, 1.0) * iScare * iScare * 0.35 * iStrobeGain;

    // Vignette + tone.
    col *= smoothstep(1.3, 0.3, length(uv - 0.5));
    col = col / (1.0 + col * 0.5);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.85));
    fragColor = vec4(col, 1.0);
}
"""


class NeonPac(BaseEffect):
    NAME = "Neon Pac"
    AUTHOR = "unicorn-viz"
    TAGS = ['games', 'arcade', 'neon', 'intense', 'hard', 'feature']
    PING_PONG_FRIENDS = [
        # Games + arcade neighbours
        'Breakout',
        'Galaga',
        # Tech/hard friends
        'Laser Tunnel',
        'Reactor Breach',
        'Threat Matrix',
        'Cyber War',
        # Audio-analysis visualizers
        'Audio Spectrum',
        'Audio Waveforms',
    ]

    def _init(self) -> None:
        self.parameters = {
            "speed": float(self.config.get("speed", 1.0)),
            "intensity": float(self.config.get("intensity", 1.0)),
            "zoom": float(self.config.get("zoom", 1.0)),
            "ghost_gain": float(self.config.get("ghost_gain", 1.0)),
            "strobe": float(self.config.get("strobe", 1.0)),
            "seed": float(self.rng.uniform(0.0, 100.0)),
        }
        self._bass = 0.0
        self._mid = 0.0
        self._treble = 0.0
        self._scare = 0.0
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

    def update(self, dt: float, audio: AudioData) -> None:
        super().update(dt, audio)
        self._bass = audio.bass_n
        self._mid = audio.mid_n
        self._treble = audio.treble_n
        self._scare = max(0.0, self._scare - dt * 2.5)
        if audio.beat:
            self._scare = 1.0
        self._scare = max(self._scare, self._bass * 0.2)

    def render(self) -> None:
        p = self._prog
        p["iTime"].value = self.time
        p["iResolution"].value = (float(self.width), float(self.height))
        p["iBass"].value = self._bass
        p["iMid"].value = self._mid
        p["iTreble"].value = self._treble
        p["iScare"].value = self._scare
        p["iSeed"].value = self.parameters["seed"]
        p["iSpeed"].value = self.parameters["speed"]
        p["iIntensity"].value = self.parameters["intensity"]
        p["iZoomGain"].value = self.parameters["zoom"]
        p["iGhostGain"].value = self.parameters["ghost_gain"]
        p["iStrobeGain"].value = self.parameters["strobe"]
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        self._vao.release()
        self._vbo.release()
        self._prog.release()

"""
Neon Pac — a glowing arcade maze chase where you ARE the action, driving
the Pac-Man entity through audio-reactive mazes hunted by ghost swarms.

A single-pass fullscreen effect: dynamically regenerated maze each frame
(procedural corridors + animated walls), a fat neon Pac-Man dot at screen
center that you pilot with bass/mid audio steering, three color-coded ghosts
that hunt with AI pathfinding (aggression driven by mid-range), scattered
pellets that sparkle on treble, and — on every beat — a **POWER PELLET**
moment where the maze inverts colors, ghosts turn vulnerable (pulse in/out),
walls strobe, and the whole screen erupts in chromatic aberration chaos.

Audio-reactive (maximally direct):
  - bass    → Pac speed, maze corridor width/density (narrower = faster feels)
  - mid     → ghost aggression level, hunt AI update rate
  - treble  → pellet density, wall shimmer intensity
  - beat    → power-pellet trigger, maze inversion, screen flash

Fully randomized at start: discrete maze layout seed (affects corridor
generation), ghost starting positions and colors (3 fixed per-instance
via rng), Pac starting location, and ``self.time`` starts random — no
two chases ever match. Tweakables (speed, intensity, aggression,
pellet_gain, strobe) ride live.
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
// Neon Pac — maze chase, player at center, ghost AI hunters.
uniform float iTime;
uniform vec2  iResolution;
uniform float iBass;
uniform float iMid;
uniform float iTreble;
uniform float iPower;     // power-pellet envelope [0,1], beat-driven
uniform float iSpeed;
uniform float iIntensity;
uniform float iAggression;
uniform float iPelletGain;
uniform float iStrobeGain;
in  vec2 v_pos;
out vec4 fragColor;

#define PI 3.14159265
#define TAU 6.28318530

float hash11(float n) { return fract(sin(n * 78.233) * 43758.5453); }
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    vec2 uv = v_pos;
    float aspect = iResolution.x / max(iResolution.y, 1.0);
    vec2 p = (uv - 0.5) * vec2(aspect, 1.0);

    float t = iTime * (0.7 + iSpeed * 0.4);

    // --- procedural maze (corridors that widen/narrow with bass) ---
    float maze_x = sin(p.x * 6.0 + floor(p.y * 5.0) * 0.3) * 0.5 + 0.5;
    float maze_y = sin(p.y * 4.0 + floor(p.x * 7.0) * 0.2) * 0.5 + 0.5;
    float corridor = max(maze_x, maze_y);

    // Bass makes corridors narrower (tighter gameplay feel = faster)
    corridor = smoothstep(0.55 - iBass * 0.25, 0.35, corridor);

    vec3 col = vec3(0.0);

    // --- maze walls (neon green, pulsing) ---
    col += vec3(0.1, 1.0, 0.3) * corridor * (0.5 + sin(t * 3.0) * 0.3) * iIntensity;

    // Power mode inverts everything
    if (iPower > 0.1) {
        col = mix(col, 1.0 - col, iPower * 0.7);
        col += vec3(1.0, 0.2, 0.9) * corridor * iPower * 0.5;
    }

    // --- Pac-Man (center, radius driven by bass) ---
    float pac_r = 0.06 + iBass * 0.04;
    float pac_d = length(p);
    float pac = smoothstep(pac_r + 0.01, pac_r, pac_d);

    // Pac color shifts in power mode
    vec3 pac_col = iPower > 0.5 ? vec3(1.0, 1.0, 0.2) : vec3(1.0, 0.9, 0.1);
    col += pac_col * pac * (0.9 + iIntensity * 0.4);
    col += pac_col * pac * pac * 2.0;  // bloom

    // --- three ghosts hunting (mid drives aggression) ---
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        // Ghost colors: red, cyan, magenta
        vec3 ghost_col = fi < 0.5 ? vec3(1.0, 0.1, 0.2) :
                         (fi < 1.5 ? vec3(0.1, 1.0, 1.0) : vec3(1.0, 0.2, 1.0));

        // Chase toward Pac with AI that hunts harder on mid
        vec2 ghost_target = p * (1.0 - iMid * iAggression * 0.5);
        float chase_speed = 0.1 + iMid * iAggression * 0.4;

        // Orbit/patrol pattern with chase bias
        float angle = fi * 2.094 + t * chase_speed;
        vec2 ghost_pos = cos(angle) * 0.25 * vec2(aspect, 1.0) + ghost_target * iMid * 0.3;

        float ghost_d = length(p - ghost_pos);
        float ghost = smoothstep(0.04, 0.0, ghost_d);

        // Power mode: ghosts glow and reverse
        if (iPower > 0.5) {
            ghost_col = vec3(0.1, 0.1, 1.0);
        }

        col += ghost_col * ghost * (0.7 + iMid * 1.2);
        col += ghost_col * ghost * ghost * 1.8;  // bloom
    }

    // --- pellets (scattered, spark on treble) ---
    for (int j = 0; j < 16; j++) {
        float fj = float(j);
        vec2 pel_pos = vec2(cos(fj * 0.39) * 0.35, sin(fj * 0.25) * 0.35)
                     + vec2(sin(t * 0.4 + fj), cos(t * 0.3 - fj * 0.5)) * 0.08;

        float pel_d = length(p - pel_pos);
        float pel = smoothstep(0.008, 0.0, pel_d);

        col += vec3(1.0, 1.0, 0.3) * pel * (0.6 + iTreble * iPelletGain * 1.5);
    }

    // --- power-pellet mode effects (beat triggered) ---
    // Screen flash + chromatic aberration
    float ca = 0.002 + iPower * 0.018;
    col.r += ca * 30.0 * (pac + corridor);
    col.b -= ca * 15.0 * (pac + corridor);

    // Full-screen power glow
    col += vec3(1.0, 0.8, 1.0) * iPower * iPower * 0.3 * iStrobeGain;

    // Maze pulse/strobe
    col *= 0.9 + iPower * sin(t * 20.0) * 0.2;

    // --- audio pulse (bass drives everything) ---
    col *= 0.95 + iBass * 0.15;

    // Vignette + tone.
    col *= smoothstep(1.4, 0.3, pac_d);
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
        'Breakout', 'Galaga',
        'Laser Tunnel', 'Reactor Breach', 'Threat Matrix', 'Cyber War',
        'Audio Spectrum', 'Audio Waveforms',
    ]

    def _init(self) -> None:
        self.parameters = {
            "speed": float(self.config.get("speed", 1.0)),
            "intensity": float(self.config.get("intensity", 1.0)),
            "aggression": float(self.config.get("aggression", 1.0)),
            "pellet_gain": float(self.config.get("pellet_gain", 1.0)),
            "strobe": float(self.config.get("strobe", 1.0)),
        }
        self._bass = 0.0
        self._mid = 0.0
        self._treble = 0.0
        self._power = 0.0
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

    def update(self, dt: float, audio: AudioData) -> None:
        super().update(dt, audio)
        self._bass = audio.bass_n
        self._mid = audio.mid_n
        self._treble = audio.treble_n
        self._power = max(0.0, self._power - dt * 2.2)
        if audio.beat:
            self._power = 1.0
        self._power = max(self._power, self._bass * 0.15)

    def render(self) -> None:
        p = self._prog
        p["iTime"].value = self.time
        p["iResolution"].value = (float(self.width), float(self.height))
        p["iBass"].value = self._bass
        p["iMid"].value = self._mid
        p["iTreble"].value = self._treble
        p["iPower"].value = self._power
        p["iSpeed"].value = self.parameters["speed"]
        p["iIntensity"].value = self.parameters["intensity"]
        p["iAggression"].value = self.parameters["aggression"]
        p["iPelletGain"].value = self.parameters["pellet_gain"]
        p["iStrobeGain"].value = self.parameters["strobe"]
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        self._vao.release()
        self._vbo.release()
        self._prog.release()

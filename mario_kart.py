"""Mario Kart — a Mode 7 circuit, driven rather than animated.

The floor is drawn the way the SNES drew it: for every pixel below the
horizon, the ray is intersected with the ground plane, rotated into the kart's
heading and looked up in the world. That single mapping is what produces the
whole effect -- a flat texture that sweeps, turns and converges correctly,
with no geometry anywhere.

The circuit is an **analytic closed curve**, ``R(theta)``, and both halves of
the effect read the same one: the shader asks "how far is this ground point
from the racing surface" to decide road, rumble strip or grass, and the
simulation asks the same function where the track goes next in order to steer.
There is no track texture to fall out of sync with the driving, because there
is no second copy of the track.

The kart is driven, not placed. It carries speed, it is steered toward a point
ahead on the circuit rather than pinned to the centreline, and the difference
between where it is pointing and where it is going is what leans it into a
corner and throws drift sparks off the inside wheels. Rivals run the same
model from their own positions, so they take the same racing line and a corner
bunches them up the way a corner does.

Audio reactivity:
  - bass    -> throttle, engine glow, camera dip
  - mid     -> palette drift, sky band
  - treble  -> drift sparks, grass detail
  - beat    -> boost: a hard shove, exhaust flare and a wider field of view

Randomised per activation: circuit shape (the harmonics of R(theta)), palette
hue, sky, grid position, and which rivals are where.

Config keys (under ``[effects.MarioKart]``):
    speed, reactivity, glitter
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.18.0'

_NRIVAL = 6
_NBOX = 8
_ROAD_HALF = 7.0        # half the racing surface's width, world units
_HORIZON = 0.60         # screen height the horizon sits at
_CAM_H = 2.6            # camera height above the plane
_CAM_BACK = 7.0         # how far behind the kart the camera sits
# A vertical object of world height h at forward distance f covers h/f of the
# screen, and the projector hands out v = _CAM_H/f. So a sprite whose local
# coordinates span H_q units tall wants scale = v * h / (H_q * _CAM_H). These
# two constants are that ratio worked out for each sprite, and they are the
# difference between a kart beside you and a kart the size of the sky.
_RIVAL_K = 1.85 / (1.55 * _CAM_H)   # kart + driver, 1.85 world units tall
_BOX_K = 1.55 / (1.80 * _CAM_H)     # item box, 1.55 units on a side
_TAU = 2.0 * math.pi
_BOX_GAP = 0.15         # radians between item boxes in the band ahead
# Sprites are painted in order with no depth test against the floor, so a kart
# on the far side of the loop would draw straight over the near grass. Nothing
# further round the circuit than this is drawn at all.
_DRAW_ARC = 1.30

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
uniform vec2  iResolution;
uniform float iTime;
uniform float iBass;
uniform float iMid;
uniform float iTreble;
uniform float iBeat;
uniform float iBoost;
uniform float iPunch;       // the beat's transient, decaying ~3x faster
uniform vec4  iBand[4];     // 16 FFT bands, low to high
uniform float iHue;
uniform float iGlitter;
uniform float iSky;
uniform vec2  iCamPos;         // camera position on the ground plane
uniform float iCamHeading;     // radians; +x at 0
uniform float iLean;           // drift lean, radians of roll
uniform vec3  iTrack;          // R0, a1, a2
uniform vec2  iTrackPhase;     // p1, p2
uniform vec4  iRival[6];       // screen x, screen y, screen scale, hue
uniform vec4  iBox[8];         // screen x, screen y, screen scale, spin
in  vec2 v_uv;
out vec4 fragColor;

#define TAU 6.28318530

float hash21(vec2 p) {
    p = mod(p, 512.0);
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

// The 16 bands, packed four to a vec4. `bandU` reads the spectrum as a curve
// across 0..1 so a silhouette or a strip can be driven straight off it.
float bandI(int i) {
    i = clamp(i, 0, 15);
    vec4 v = iBand[i >> 2];
    int k = i & 3;
    return k == 0 ? v.x : (k == 1 ? v.y : (k == 2 ? v.z : v.w));
}
float bandU(float u) {
    float f = clamp(u, 0.0, 1.0) * 15.0;
    int i = int(f);
    return mix(bandI(i), bandI(i + 1), fract(f));
}


// Distance from a ground point to the circuit's centreline. The curve is
// star-shaped in theta, so this is exact and costs three sines -- no track
// texture, and the simulation steers off this same function.
float trackDist(vec2 w) {
    float r = length(w);
    float th = atan(w.y, w.x);
    float R = iTrack.x * (1.0 + iTrack.y * sin(th * 2.0 + iTrackPhase.x)
                              + iTrack.z * sin(th * 3.0 + iTrackPhase.y));
    return abs(r - R);
}

// A billboard: flat, scaled by its distance, drawn at a screen position the
// CPU already projected.
float sprite(vec2 p, vec2 c, vec2 half_) {
    vec2 d = abs(p - c) - half_;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float t = mod(iTime, 600.0);
    vec2 uv = v_uv;
    // A boost widens the view; the lean rolls it into the corner.
    float fov = 1.0 + iBoost * 0.16 + iPunch * 0.045;
    vec2 p0 = (uv - 0.5) * vec2(A, 1.0) * fov;
    // The lean rolls the world. The kart is bolted to the camera, so it is
    // drawn in p0 -- roll it too and it swings around the frame instead of
    // leaning into the corner.
    float cs = cos(iLean), sn = sin(iLean);
    vec2 p = mat2(cs, -sn, sn, cs) * p0;
    float horizon = (0.60 - 0.5) + iBass * 0.012;

    vec3 neon = palette(iHue);
    vec3 col;

    if (p.y >= horizon) {
        // ---- sky ----
        float up = clamp((p.y - horizon) / (0.5 - horizon + 1e-3), 0.0, 1.0);
        vec3 lo, hi;
        if (iSky < 0.34)      { lo = vec3(0.55, 0.80, 1.00); hi = vec3(0.10, 0.32, 0.80); }
        else if (iSky < 0.67) { lo = vec3(1.00, 0.70, 0.38); hi = vec3(0.20, 0.16, 0.56); }
        else                  { lo = vec3(0.30, 0.20, 0.58); hi = vec3(0.05, 0.06, 0.26); }
        col = mix(lo, hi, pow(up, 0.85));
        col = mix(col, palette(iHue + 0.5), 0.10 + iMid * 0.10);
        // A sun low over the circuit: disc, bloom, and a bar of haze where it
        // meets the horizon.
        vec2 sunP = vec2(sin(iCamHeading * 0.5) * 0.55, horizon + 0.20);
        float sd = length((p - sunP) * vec2(1.0, 1.15));
        vec3 sunCol = mix(vec3(1.0, 0.88, 0.55), palette(iHue + 0.08), 0.35);
        col += sunCol * exp(-sd * 5.5) * (0.55 + iMid * 0.45);
        col = mix(col, sunCol, smoothstep(0.085, 0.062, sd));
        // An aurora over the circuit, reading the spectrum left to right.
        // The sky was the largest surface in the frame and moved with nothing.
        float au = clamp(p.x / A + 0.5, 0.0, 1.0);
        float spec = bandU(au);
        float curt = smoothstep(0.0, 0.55 + spec * 0.85, p.y - horizon);
        col += mix(palette(iHue + 0.22), vec3(1.0), 0.25)
             * (1.0 - curt) * smoothstep(0.0, 0.05, p.y - horizon)
             * (spec * 1.25 + iPunch * 0.30);
        // Clouds banded along the horizon, and distant hills under them.
        float band = sin(p.x * 3.1 + iCamHeading * 2.0) * 0.5 + 0.5;
        col = mix(col, vec3(1.0), smoothstep(0.55, 0.95, band) * (1.0 - up) * 0.35);
        float hill = 0.045 * (sin(p.x * 2.2 + iCamHeading * 1.6) + 0.6 * sin(p.x * 5.1));
        if (p.y < horizon + 0.05 + hill) {
            col = mix(col, mix(vec3(0.12, 0.40, 0.22), palette(iHue + 0.3), 0.20), 0.85);
        }
    } else {
        // ---- Mode 7 ----
        // Intersect with the ground plane, then rotate into the heading. This
        // one mapping is the whole trick: everything on the floor follows from
        // it, and nothing on the floor is geometry.
        float v = horizon - p.y;
        float z = 2.6 / max(v, 1e-4);
        vec2 cam = vec2(p.x * z, z);
        float ch = cos(iCamHeading), sh = sin(iCamHeading);
        // Right-handed: forward is the heading, screen-right is heading
        // rotated -90 deg. Getting this sign wrong mirrors the world, and the
        // lean then rolls the camera into the outside of every corner.
        vec2 w = iCamPos + vec2(cam.y * ch + cam.x * sh,
                                cam.y * sh - cam.x * ch);

        float d = trackDist(w);
        float th = atan(w.y, w.x);
        float fade = clamp(1.0 - z / 150.0, 0.0, 1.0);

        // Grass, with a check pattern that gives the speed something to read
        // against -- a flat green floor reads as standing still.
        float check = mod(floor(w.x * 0.25) + floor(w.y * 0.25), 2.0);
        vec3 grass = mix(vec3(0.14, 0.58, 0.21), vec3(0.19, 0.70, 0.27), check);
        grass = mix(grass, palette(iHue + 0.3), 0.08);
        col = grass;

        if (d < 7.0 + 2.2) {
            // Rumble strip: red/white blocks stepping along the edge.
            // ~4 world units per block at this radius. At the 26.0 this
            // started from, one stripe ran 27 units and read as a paint job.
            float stripe = mod(floor(th * 170.0), 2.0);
            vec3 rumble = mix(vec3(0.92, 0.18, 0.16), vec3(0.96, 0.96, 0.96), stripe);
            col = mix(col, rumble, smoothstep(7.0 + 2.2, 7.0 + 1.4, d));
        }
        if (d < 7.0) {
            // Road: asphalt with a dashed centre line.
            vec3 road = mix(vec3(0.24, 0.24, 0.28), palette(iHue + 0.6), 0.06);
            road *= 0.88 + 0.24 * hash21(floor(w * 3.0));
            float centre = smoothstep(0.55, 0.0, d);
            float dash = step(0.5, mod(floor(th * 60.0), 2.0));
            road = mix(road, vec3(0.95, 0.92, 0.55), centre * dash * 0.8);
            col = mix(col, road, smoothstep(7.0, 7.0 - 0.9, d));
            // Neon kerb light where the road meets the rumble.
            float ku = clamp(p.x / A + 0.5, 0.0, 1.0);
            col += neon * smoothstep(0.55, 0.0, abs(d - 7.0))
                 * (0.28 + iBass * 1.3 + bandU(ku) * 1.5 + iPunch * 0.9);
            // Start/finish: two rows of checks laid across the lap origin,
            // sized in world units so the squares stay square.
            float sf = smoothstep(0.050, 0.040, abs(th));
            float ck = mod(floor(d * 0.85) + floor(abs(th) * 45.0), 2.0);
            col = mix(col, vec3(0.04 + ck * 0.90), sf * 0.95);
        }
        // Distance haze into the sky, or the far floor stacks up flat.
        vec3 haze = mix(vec3(0.55, 0.72, 0.95), palette(iHue + 0.5), 0.25);
        col = mix(haze, col, fade);
        // Speed streaks rushing under the kart.
        col += vec3(1.0) * step(0.997, hash21(floor(vec2(w.x * 2.0, w.y * 2.0 + t * 40.0))))
             * fade * iBoost * 0.5;
        // Glitter coming up off the surface: round points inside their cells,
        // never a bare threshold on the cell hash, which lights whole squares.
        vec2 gc = vec2(w.x * 1.7, w.y * 1.7 + t * 9.0);
        vec2 gf = fract(gc) - 0.5;
        col += mix(vec3(1.0), neon, 0.35)
             * step(0.962 - iPunch * 0.030, hash21(floor(gc)))
             * smoothstep(0.34, 0.02, length(gf)) * fade
             * (0.25 + iTreble * 1.6 + iPunch * 1.2) * iGlitter;
    }

    // ---- item boxes ----
    for (int i = 0; i < 8; i++) {
        vec4 B = iBox[i];
        if (B.z <= 0.0) continue;
        vec2 q = (p - B.xy) / B.z;
        if (abs(q.x) > 1.6 || abs(q.y) > 1.6) continue;
        // A spinning cube reads as a rotating diamond with a bright core.
        float sp = B.w;
        vec2 r = mat2(cos(sp), -sin(sp), sin(sp), cos(sp)) * q;
        float dia = abs(r.x) + abs(r.y) - 0.9;
        vec3 boxCol = palette(iHue + 0.15 + 0.25 * sin(sp));
        col = mix(col, boxCol, smoothstep(0.06, 0.0, dia) * 0.75);
        col += boxCol * exp(-max(dia, 0.0) * 5.0)
             * (0.5 + iTreble * 1.6 + iPunch * 1.2) * iGlitter;
        col += vec3(1.0) * smoothstep(0.22, 0.0, length(q)) * 0.5;
    }

    // ---- rivals ----
    for (int i = 0; i < 6; i++) {
        vec4 R = iRival[i];
        if (R.z <= 0.0) continue;
        vec2 q = (p - R.xy) / R.z;
        if (abs(q.x) > 1.8 || abs(q.y) > 1.6) continue;
        vec3 body = palette(R.w);
        // Kart from behind: body, two tyres, a head over the wheel.
        float chassis = sprite(q, vec2(0.0, 0.18), vec2(0.62, 0.24));
        float tyres = min(sprite(q, vec2(-0.72, 0.12), vec2(0.20, 0.30)),
                          sprite(q, vec2(0.72, 0.12), vec2(0.20, 0.30)));
        // A helmet, not a bare head: at this size a skin-coloured disc just
        // reads as a balloon floating over the kart, and rivals routinely
        // show only their head above the kart in front.
        float helm = length(q - vec2(0.0, 0.64)) - 0.27;
        float visor = max(helm + 0.07, abs(q.y - 0.60) - 0.08);
        col = mix(col, vec3(0.07), smoothstep(0.05, 0.0, tyres));
        col = mix(col, body, smoothstep(0.05, 0.0, chassis));
        col = mix(col, mix(body, vec3(1.0), 0.35), smoothstep(0.05, 0.0, helm));
        col = mix(col, vec3(0.10, 0.12, 0.16), smoothstep(0.04, 0.0, visor));
        col += body * smoothstep(0.09, 0.0, abs(chassis)) * 0.4;
    }

    // ---- the player's kart, held at the bottom of the frame ----
    // Seen from behind: the tyres, the chassis, the back of the driver. The
    // scale is set so he covers about a third of the width -- enough to read
    // as a character, little enough to leave the corner visible.
    {
        vec2 q = (p0 - vec2(0.0, -0.36)) / 0.125;
        if (abs(q.x) < 2.6 && abs(q.y) < 2.6) {
            vec3 red = vec3(0.90, 0.14, 0.12);
            vec3 body = mix(red, palette(iHue), 0.12);
            vec3 skin = vec3(1.00, 0.80, 0.60);

            float tyres = min(sprite(q, vec2(-1.18, -0.10), vec2(0.30, 0.44)),
                              sprite(q, vec2(1.18, -0.10), vec2(0.30, 0.44)));
            float chassis = sprite(q, vec2(0.0, 0.02), vec2(0.95, 0.40));
            float seat = sprite(q, vec2(0.0, 0.52), vec2(0.52, 0.30));
            float torso = sprite(q, vec2(0.0, 0.86), vec2(0.46, 0.34));
            float head = length(q - vec2(0.0, 1.42)) - 0.38;
            float capDome = length(q - vec2(0.0, 1.54)) - 0.42;
            float capBrim = sprite(q, vec2(0.0, 1.22), vec2(0.46, 0.09));
            // Gloves out on the wheel, and the wheel itself behind them.
            float wheel = abs(length(q - vec2(0.0, 0.74)) - 0.40) - 0.07;
            float glove = min(length(q - vec2(-0.40, 0.80)) - 0.16,
                              length(q - vec2(0.40, 0.80)) - 0.16);

            col = mix(col, vec3(0.06, 0.06, 0.07), smoothstep(0.05, 0.0, tyres));
            col = mix(col, body, smoothstep(0.05, 0.0, chassis));
            col = mix(col, body * 0.72, smoothstep(0.05, 0.0, seat));
            col = mix(col, vec3(0.14), smoothstep(0.05, 0.0, wheel));
            // Blue overall straps crossing the red shirt.
            vec3 shirt = red;
            if (abs(abs(q.x) - 0.26) < 0.11) shirt = vec3(0.16, 0.32, 0.82);
            col = mix(col, shirt, smoothstep(0.05, 0.0, torso));
            col = mix(col, vec3(0.96), smoothstep(0.05, 0.0, glove));
            // Hair showing under the cap, then the cap over it.
            col = mix(col, vec3(0.30, 0.16, 0.07),
                      smoothstep(0.05, 0.0, head));
            col = mix(col, skin, smoothstep(0.05, 0.0,
                      max(head, q.y - 1.30)) * 0.55);
            col = mix(col, red, smoothstep(0.05, 0.0, min(capDome, capBrim)));
            col += body * smoothstep(0.10, 0.0, abs(chassis)) * 0.35;

            // Exhaust: flares hard on a boost.
            for (int s = -1; s <= 1; s += 2) {
                vec2 e = q - vec2(0.62 * float(s), -0.44);
                col += mix(vec3(1.0, 0.66, 0.20), neon, 0.35)
                     * exp(-dot(e, e) * 7.0)
                     * (0.20 + iBoost * 2.0 + iBass * 0.80 + iPunch * 1.3);
            }
        }
        // Drift sparks off the inside wheels, thrown by the lean.
        float lean = clamp(abs(iLean) * 4.5, 0.0, 1.0);
        for (int i = 0; i < 9; i++) {
            float fi = float(i);
            float ph = fract(t * 2.4 + fi * 0.111);
            vec2 sp2 = vec2(-sign(iLean) * (0.15 + ph * 0.40),
                            -0.46 + ph * 0.14 + sin(fi * 2.0) * 0.02);
            vec2 dv = p0 - sp2;
            float q2 = dot(dv, dv);
            if (q2 < 0.004) {
                col += mix(vec3(1.0, 0.85, 0.35), neon, 0.3)
                     * exp(-q2 * 1400.0) * (1.0 - ph) * lean
                     * (0.55 + iTreble * 1.9 + iPunch * 1.0) * iGlitter;
            }
        }
    }

    col += neon * (iBeat * 0.07 + iPunch * 0.11);
    col *= smoothstep(1.25, 0.32, length((uv - 0.5) * vec2(A, 1.0)));
    col = col / (1.0 + col * 0.34);
    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.86)), 1.0);
}
"""


class MarioKart(BaseEffect):
    """Mode 7 circuit racing: one analytic track, driven by a real kart."""

    NAME = 'Mario Kart'
    AUTHOR = 'unicorn-viz'
    TAGS = ['games', 'arcade', 'neon', 'energetic', 'intense', 'hard']
    PING_PONG_FRIENDS = [
        'Super Mario', 'Donkey Kong', 'Galaga', 'Neon Pac', 'Breakout',
        'Audio Spectrum',
    ]

    _BASE_SPEED = 26.0      # world units / s at rest

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.5)),
            'glitter': float(self.config.get('glitter', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._punch = 0.0
        self._band = np.zeros(16, dtype=np.float64)
        # The spectrum falls away with frequency, so an untilted reading only
        # ever moves at its left-hand end and half the strip looks dead.
        self._band_tilt = np.linspace(1.0, 2.7, 16)
        self._tbl_band = np.zeros((4, 4), dtype='f4')
        self._boost = 0.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._sky = float(self.rng.uniform(0.0, 1.0))
        self._glitter = max(0.0, float(self.parameters['glitter']))

        # The circuit. Amplitudes stay modest so R(theta) never doubles back on
        # itself -- a star-shaped curve is what makes the distance test exact.
        self._r0 = float(self.rng.uniform(85.0, 115.0))
        self._a1 = float(self.rng.uniform(0.10, 0.20))
        self._a2 = float(self.rng.uniform(0.04, 0.10))
        self._p1 = float(self.rng.uniform(0.0, _TAU))
        self._p2 = float(self.rng.uniform(0.0, _TAU))

        self._theta = float(self.rng.uniform(0.0, _TAU))
        self._off = 0.0
        self._heading = 0.0
        self._lean = 0.0
        self._speed = self._BASE_SPEED

        # A pack, not a parade: the nearest sits a few world units ahead and
        # they step back about 12 each, so three or four are always large
        # enough to race against and the rest trail off toward the horizon.
        self._rivals = np.zeros((_NRIVAL, 4), dtype=np.float64)  # theta off hue pace
        for i in range(_NRIVAL):
            self._rivals[i] = (
                (self._theta + 0.07 + i * 0.115
                 + self.rng.uniform(0.0, 0.07)) % _TAU,
                self.rng.uniform(-_ROAD_HALF * 0.62, _ROAD_HALF * 0.62),
                self.rng.uniform(0.0, 1.0),
                self.rng.uniform(0.90, 1.06),
            )
        # Boxes live in a band ahead and are recycled to the back of it, so
        # there is always a row coming at you without any of them sitting on
        # the far side of the loop.
        self._boxes = np.zeros((_NBOX, 2), dtype=np.float64)        # theta off
        for i in range(_NBOX):
            self._boxes[i] = ((self._theta + 0.10 + i * _BOX_GAP) % _TAU,
                              (i % 3 - 1) * _ROAD_HALF * 0.45)
        self._spin = 0.0

        self._tbl_rival = np.zeros((_NRIVAL, 4), dtype='f4')
        self._tbl_box = np.zeros((_NBOX, 4), dtype='f4')
        self._aspect = 16.0 / 9.0

    # ------------------------------------------------------------- geometry
    def _radius(self, th: float) -> float:
        """The circuit's radius at a heading angle -- the shader's own R()."""
        return self._r0 * (1.0 + self._a1 * math.sin(th * 2.0 + self._p1)
                                + self._a2 * math.sin(th * 3.0 + self._p2))

    def _point(self, th: float, off: float = 0.0) -> tuple[float, float]:
        r = self._radius(th) + off
        return (r * math.cos(th), r * math.sin(th))

    def _project(self, wx: float, wy: float, cam: tuple[float, float],
                 heading: float) -> tuple[float, float, float]:
        """World point -> screen, using the inverse of the shader's Mode 7.

        Returns screen x, screen y and the depth scale; depth <= 0 means the
        point is behind the camera and must not be drawn.
        """
        dx, dy = wx - cam[0], wy - cam[1]
        ch, sh = math.cos(heading), math.sin(heading)
        # Camera space, exactly inverting the shader's mapping: forward along
        # the heading, right along the heading rotated -90 degrees.
        fwd = dx * ch + dy * sh
        side = dx * sh - dy * ch
        if fwd <= 1.2:
            return (0.0, 0.0, -1.0)
        v = _CAM_H / fwd
        return (side / fwd, (_HORIZON - 0.5) - v, v)

    # ------------------------------------------------------------------ sim
    def _pump_bands(self, audio: AudioData, dt: float) -> None:
        """Fold the FFT down to 16 bands the shader can read as a curve."""
        b = getattr(audio, 'bands', None)
        if b is not None and len(b) >= 16:
            edges = np.linspace(0, len(b), 17).astype(int)
            tgt = np.array([float(np.mean(b[edges[i]:max(edges[i] + 1, edges[i + 1])]))
                            for i in range(16)]) * self._band_tilt
            self._band += (tgt - self._band) * min(1.0, dt * 14.0)
        else:
            self._band *= math.exp(-dt * 2.5)
        np.clip(self._band, 0.0, 1.6, out=self._band)
        self._tbl_band[:] = self._band.reshape(4, 4).astype('f4')

    def update(self, dt: float, audio: AudioData) -> None:
        """Drive the lap: steer toward the track ahead, lean into what that costs."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 1.8))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 1.8))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 1.8))
        self._pump_bands(audio, dt)
        if audio.beat > 0.5:
            self._beat = 1.0
            self._boost = 1.0
            self._punch = 1.0
        self._beat = max(0.0, self._beat - dt * 3.4)
        # A second, much faster envelope. iBeat at 3.4/s is a ~300 ms swell,
        # which reads as a pulse; a transient needs to be gone again before
        # the next one lands.
        self._punch = max(0.0, self._punch - dt * 11.0)
        self._boost = max(0.0, self._boost - dt * 1.6)

        throttle = max(0.05, float(self.parameters['speed']))
        target = self._BASE_SPEED * throttle * (0.85 + self._bass * 0.55
                                                + self._boost * 0.55)
        self._speed += (target - self._speed) * min(1.0, dt * 2.0)

        # Advance along the circuit. d(theta) = v*dt / r, so the kart covers
        # ground at a real speed rather than a constant angular rate -- the
        # difference shows on every tightening corner.
        r = max(self._radius(self._theta), 1.0)
        prev = self._theta
        self._theta = (self._theta + self._speed * dt / r) % _TAU

        # The racing line: aim off-centre, wide on entry and tight at the apex.
        line = math.sin(self._theta * 2.0 + self._p1) * _ROAD_HALF * 0.45
        self._off += (line - self._off) * min(1.0, dt * 1.4)

        # Heading from where the kart actually went, and lean from how fast
        # that heading is turning.
        x0, y0 = self._point(prev, self._off)
        x1, y1 = self._point(self._theta, self._off)
        if (x1 - x0) or (y1 - y0):
            want = math.atan2(y1 - y0, x1 - x0)
            turn = math.atan2(math.sin(want - self._heading),
                              math.cos(want - self._heading))
            self._heading = want
            # 0.45 rad of roll per rad/s of yaw. A lap turns the heading at
            # roughly 0.35-0.9 rad/s, so a corner lands around 15-20 degrees
            # and the tightest apexes hit the clamp -- at the 0.10 this
            # started at, the whole lap sat under 3 degrees and the drift
            # sparks, which gate on the lean, never fired at all.
            lean_target = float(np.clip(turn / max(dt, 1e-3) * 0.45, -0.34, 0.34))
            self._lean += (lean_target - self._lean) * min(1.0, dt * 4.0)

        # Each rival holds its own pace, so the field stretches and bunches
        # over a lap instead of moving as one rigid block.
        self._rivals[:, 0] = (self._rivals[:, 0]
                              + self._speed * dt / r * self._rivals[:, 3]) % _TAU
        self._spin += dt * 3.0
        # Recycle any box the kart has gone past to the back of the band.
        dth = (self._boxes[:, 0] - self._theta) % _TAU
        passed = dth > _TAU - 0.15
        if passed.any():
            self._boxes[passed, 0] = (self._theta
                                      + _NBOX * _BOX_GAP) % _TAU
            self._boxes[passed, 1] = self.rng.choice(
                (-1.0, 0.0, 1.0), size=int(passed.sum())) * _ROAD_HALF * 0.45

    def render(self) -> None:
        """Project the field with the inverse of the shader's own mapping."""
        self._aspect = float(self.width) / max(float(self.height), 1.0)
        kx, ky = self._point(self._theta, self._off)
        # Camera sits behind the kart along its heading.
        cam = (kx - math.cos(self._heading) * _CAM_BACK,
               ky - math.sin(self._heading) * _CAM_BACK)

        self._tbl_rival[:] = 0.0
        shown = []
        for i in range(_NRIVAL):
            th, off, hue = self._rivals[i, 0], self._rivals[i, 1], self._rivals[i, 2]
            wx, wy = self._point(float(th), float(off))
            if (float(th) - self._theta) % _TAU > _DRAW_ARC:
                continue
            sx, sy, v = self._project(wx, wy, cam, self._heading)
            if v > 0.0:
                shown.append((v, sx, sy, float(hue)))
        # Far to near, so a kart you are catching draws over the one ahead.
        shown.sort()
        for i, (v, sx, sy, hue) in enumerate(shown[:_NRIVAL]):
            self._tbl_rival[i] = (sx, sy, max(v * _RIVAL_K, 0.004), hue)
        self._tbl_box[:] = 0.0
        for i in range(_NBOX):
            th, off = self._boxes[i]
            if (float(th) - self._theta) % _TAU > _DRAW_ARC:
                continue
            wx, wy = self._point(float(th), float(off))
            sx, sy, v = self._project(wx, wy, cam, self._heading)
            if v > 0.0:
                self._tbl_box[i] = (sx, sy + v * 0.46, max(v * _BOX_K, 0.003),
                                    float(self._spin + i))

        p = self._prog
        p['iResolution'].value = (float(self.width), float(self.height))
        p['iTime'].value = float(self.time)
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iPunch'].value = self._punch
        p['iBand'].write(self._tbl_band.tobytes())
        p['iBoost'].value = self._boost
        p['iHue'].value = self._hue
        p['iGlitter'].value = self._glitter
        p['iSky'].value = self._sky
        p['iCamPos'].value = (float(cam[0]), float(cam[1]))
        p['iCamHeading'].value = float(self._heading)
        p['iLean'].value = float(self._lean)
        p['iTrack'].value = (float(self._r0), float(self._a1), float(self._a2))
        p['iTrackPhase'].value = (float(self._p1), float(self._p2))
        p['iRival'].write(self._tbl_rival.tobytes())
        p['iBox'].write(self._tbl_box.tobytes())
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()

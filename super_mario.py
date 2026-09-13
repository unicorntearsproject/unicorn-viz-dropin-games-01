"""Super Mario — a real auto-running platformer, simulated then drawn.

Like the rest of this pack (see ``breakout.py`` for the pattern this follows),
nothing here is an animation loop. There is a level, a runner with mass, and
the runner plays it: the level is generated ahead of the camera and recycled
behind it, the runner carries velocity and gravity, and every jump is a
decision taken from what is actually in front of him -- a pit he would fall
into, a goomba he would walk into, a block he could open. He clears the gaps
because he reads them, not because a curve says jump here.

The level is a **tile heightmap**, generated one column at a time on the CPU
and handed to the shader as an array. That keeps the terrain honest -- a pit is
a column with no ground, and the runner's collision, the camera and the drawn
world all read the same numbers -- and it keeps the fragment shader to looking
up a column and shading it.

What the simulation actually runs:
  - Gravity, a jump impulse, and a variable-height jump (held longer on a
    stronger beat), with coyote-time so a jump taken a frame late still fires.
  - Goombas walking left; landing on one squashes it, walking into one costs
    the run a stumble.
  - Question blocks that open when hit from below, popping a coin that arcs.
  - Pipes he has to clear, and staircases he runs up.

Audio reactivity:
  - bass    -> run speed, jump strength, ground glow
  - mid     -> parallax drift, palette
  - treble  -> coin sparkle, brick grain
  - beat    -> a jump when one is available, block pop, flash

Randomised per activation: palette hue, level seed (so the terrain, pipes and
enemies differ every run), starting distance, and time-of-day sky.

Config keys (under ``[effects.SuperMario]``):
    speed, reactivity, glitter
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.18.0'

_COLS = 19              # ground columns handed to the shader (view + margin)
_VIEW_W = 15.0          # tiles visible across the screen
_NBLOCK = 10
_NENEMY = 8
_NCOIN = 10
_NPART = 28             # event sparks: stomps, coin pops, block hits, landings


def _smoothstep(a: float, b: float, x: float) -> float:
    """GLSL smoothstep, for the biome crossfade."""
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)

_GROUND_Y = 2.0         # default ground top, in tiles from the bottom
_PIT = -5.0             # a column with no ground at all
_GRAVITY = 62.0         # tiles / s^2
_OFF = -99.0
_HEAD = 1.05            # tiles from the runner's feet to the top of his cap

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
uniform float iPunch;          // the beat's transient, decaying ~3x faster
uniform vec4  iBand[4];        // 16 FFT bands, low to high
uniform float iHue;
uniform float iGlitter;
uniform float iSky;            // per-activation time of day
uniform float iZone;           // biome index, fractional across a crossfade
uniform vec2  iFlag;           // zone-boundary flagpole: x, height (-99 = none)
uniform float iCamFrac;        // camera x within its tile, 0..1
uniform float iCamY;           // camera lift, tiles -- follows a high jump
uniform float iViewH;          // tiles visible vertically
uniform float iGround[19];     // ground top per column, tiles (_PIT = no ground)
uniform vec4  iBlock[10];      // x, y, kind (0 brick 1 question 2 spent), hit
uniform vec4  iEnemy[8];       // x, y, squash (0 alive .. 1 flat), face
uniform vec4  iEnemyK[8];      // kind (0 goomba 1 koopa 2 piranha), walk, rise, _
uniform vec4  iCoin[10];       // x, y, life, spin
uniform vec4  iPart[28];       // x, y, life 1..0, kind (0 dust 1 gold 2 stomp)
uniform vec4  iPipe[4];        // x, height, _, _
uniform vec3  iHero;           // x, y (feet), face
uniform float iHeroAir;        // 0 grounded, 1 airborne
uniform float iHeroStumble;
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
float box(vec2 p, vec2 b) { vec2 d = abs(p) - b; return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0); }

// The 16 bands, packed four to a vec4. `bandU` reads the spectrum as a curve
// across 0..1 so a silhouette can be driven straight off it.
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

// Ground top at a world column, with the camera's sub-tile offset applied.
float groundAt(float wx) {
    int i = int(clamp(floor(wx), 0.0, 18.0));
    return iGround[i];
}

// Six biomes. The run crosses one every ninety tiles and crossfades over the
// last few, so the level is not one palette for longer than it is on screen --
// which is what "a bit boring" actually was.
struct Biome {
    vec3 skyLo; vec3 skyHi; vec3 hill; vec3 soil; vec3 cap;
    float hills; float clouds; float stars; float snow;
};
Biome biomeOf(int z) {
    if (z == 1) return Biome(vec3(1.00, 0.62, 0.30), vec3(0.22, 0.12, 0.44),
                             vec3(0.20, 0.34, 0.40), vec3(0.60, 0.30, 0.17),
                             vec3(0.30, 0.66, 0.30), 1.0, 1.0, 0.0, 0.0);
    if (z == 2) return Biome(vec3(0.16, 0.20, 0.48), vec3(0.03, 0.04, 0.18),
                             vec3(0.10, 0.20, 0.32), vec3(0.34, 0.28, 0.40),
                             vec3(0.20, 0.48, 0.40), 1.0, 0.3, 1.0, 0.0);
    if (z == 3) return Biome(vec3(0.76, 0.88, 0.98), vec3(0.38, 0.56, 0.80),
                             vec3(0.78, 0.85, 0.94), vec3(0.42, 0.47, 0.58),
                             vec3(0.92, 0.96, 1.00), 1.0, 1.0, 0.0, 1.0);
    if (z == 4) return Biome(vec3(0.09, 0.08, 0.15), vec3(0.02, 0.02, 0.06),
                             vec3(0.07, 0.06, 0.12), vec3(0.20, 0.25, 0.50),
                             vec3(0.18, 0.52, 0.74), 0.0, 0.0, 0.6, 0.0);
    if (z == 5) return Biome(vec3(0.99, 0.86, 0.55), vec3(0.52, 0.60, 0.82),
                             vec3(0.80, 0.62, 0.34), vec3(0.78, 0.62, 0.33),
                             vec3(0.86, 0.74, 0.42), 1.0, 0.5, 0.0, 0.0);
    return Biome(vec3(0.55, 0.80, 1.00), vec3(0.10, 0.30, 0.80),
                 vec3(0.13, 0.58, 0.26), vec3(0.58, 0.31, 0.16),
                 vec3(0.28, 0.80, 0.28), 1.0, 1.0, 0.0, 0.0);
}
Biome blendBiome(float zf) {
    int z0 = int(mod(floor(zf), 6.0));
    int z1 = int(mod(floor(zf) + 1.0, 6.0));
    float f = fract(zf);
    Biome a = biomeOf(z0), b = biomeOf(z1);
    return Biome(mix(a.skyLo, b.skyLo, f), mix(a.skyHi, b.skyHi, f),
                 mix(a.hill, b.hill, f), mix(a.soil, b.soil, f),
                 mix(a.cap, b.cap, f), mix(a.hills, b.hills, f),
                 mix(a.clouds, b.clouds, f), mix(a.stars, b.stars, f),
                 mix(a.snow, b.snow, f));
}

// Sky, hills and clouds. Everything here parallaxes slower than the level,
// which is what gives the run its depth.
vec3 backdrop(vec2 w, float t, Biome B) {
    float up = clamp(w.y / iViewH, 0.0, 1.0);
    // iSky keeps a per-activation shift on top of the biome, so two runs
    // through the same zone still do not match.
    vec3 col = mix(B.skyLo, B.skyHi, pow(up, 0.65 + iSky * 0.35));
    // Tint proportionally to what the biome is already doing. A flat mix
    // toward a saturated palette colour swamps the dark biomes outright --
    // underground came out as a sheet of dark red.
    float lum = dot(col, vec3(0.333));
    col = mix(col, palette(iHue + 0.55) * (0.35 + lum * 1.7), 0.16 + iMid * 0.16);
    if (B.stars > 0.01) {
        // A square cell, in world units. At 0.22 x 0.9 the cell was four times
        // wider than tall, so a round dot in cell space came out as a wide
        // lozenge on screen and the stars read as floating pills.
        vec2 sc = vec2(w.x, w.y) * 2.1;
        vec2 sf = fract(sc) - 0.5;
        col += vec3(0.85, 0.90, 1.0) * B.stars * up
             * step(0.955, hash21(floor(sc))) * smoothstep(0.30, 0.02, length(sf))
             * (0.5 + iTreble * 1.2 + iPunch * 0.8);
    }
    col += palette(iHue + 0.2) * iPunch * 0.09 * (1.0 - up * 0.5);

    // The spectrum, read left to right across the frame. The hills were the
    // largest thing on screen and the one thing that never moved with the
    // music, which is most of why only 3% of pixels reacted at all.
    float su = clamp((w.x - iCamFrac) / 15.0, 0.0, 1.0);
    float spec = bandU(su);

    // Rolling hills, two layers at different parallax rates.
    for (int k = 0; k < 2; k++) {
        float fk = float(k);
        float rate = 0.10 + fk * 0.13;
        float hx = w.x * rate + t * 0.0 + fk * 7.3;
        // Kept small: the ridge is a skyline, and a lift big enough to
        // shove it into half the frame stops being a spectrum and starts
        // being terrain that ate the level.
        float lift = spec * (0.42 + fk * 0.62);
        float top = iCamY * 0.25 + 2.3 + fk * 1.15 + lift
                  + sin(hx * 0.55) * 0.7 + sin(hx * 1.31 + 1.7) * 0.35;
        if (w.y < top && B.hills > 0.5) {
            vec3 hc = mix(B.hill, palette(iHue + 0.3), 0.08 + fk * 0.06);
            // A lighter crown along the ridge, or a hill is a flat silhouette.
            float crown = smoothstep(0.60, 0.0, top - w.y);
            col = mix(col, hc * (0.50 + fk * 0.34) * (1.0 + crown * 0.80), 0.95);
        }
        // A lit edge on the ridge, brightest where that band is loudest, so
        // the skyline reads as a spectrum and not just as terrain.
        col += mix(palette(iHue + 0.15), vec3(1.0), 0.35)
             * smoothstep(0.13, 0.0, abs(w.y - top)) * B.hills
             * (0.14 + spec * 1.10 + iPunch * 0.45) * (0.55 + fk * 0.45);
    }
    // Clouds.
    float cy = w.y - (iCamY + iViewH * 0.72);
    float cx = w.x * 0.28 + 4.0;
    float puff = exp(-pow(cy, 2.0) * 1.6)
               * smoothstep(0.35, 0.9, sin(cx * 0.9) * 0.5 + 0.5 + sin(cx * 2.3) * 0.2);
    col = mix(col, vec3(1.0), clamp(puff, 0.0, 1.0) * B.clouds * (0.48 + iMid * 0.25));
    if (B.snow > 0.01) {
        for (int k = 0; k < 2; k++) {
            float fk2 = float(k);
            vec2 sc = vec2(w.x * 2.2 + sin(w.y * 0.5 + t) * 0.5 + fk2 * 3.1,
                           (w.y + t * (0.9 + fk2 * 0.7)) * 2.2);
            vec2 sf = fract(sc) - 0.5;
            col += vec3(1.0) * step(0.958, hash21(floor(sc)))
                 * smoothstep(0.34, 0.05, length(sf)) * B.snow
                 * (0.45 + iTreble * 0.7) * (0.7 - fk2 * 0.25);
        }
    }

    // Sparkle in the air: round points inside their cells, drifting up. A
    // bare threshold on the cell hash would light whole squares.
    vec2 cell = vec2(w.x * 2.2, (w.y - t * 0.9) * 2.2);
    vec2 cf = fract(cell) - 0.5;
    float gl = step(0.986 - iPunch * 0.022, hash21(floor(cell)))
             * smoothstep(0.26, 0.02, length(cf));
    col += mix(vec3(1.0), palette(iHue + 0.35), 0.35) * gl
         * (0.50 + iTreble * 2.0 + iPunch * 1.5) * iGlitter;
    return col;
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float t = mod(iTime, 600.0);
    // World space in tiles: x runs from the camera's tile, y from the ground.
    vec2 w = vec2(iCamFrac + v_uv.x * 15.0, iCamY + v_uv.y * iViewH);

    Biome BM = blendBiome(iZone);
    vec3 col = backdrop(w, t, BM);
    vec3 neon = palette(iHue);

    // ---- the level ----
    float g = groundAt(w.x);
    if (g > (-5.0) && w.y < g) {
        float depth = g - w.y;
        vec3 soil = mix(BM.soil, palette(iHue + 0.62), 0.07);
        // Brick coursing, offset every other row -- the top course is the
        // lighter capping row the originals always had.
        vec2 bc = vec2(w.x, w.y);
        float row = floor(bc.y * 2.0);
        float bx = fract(bc.x * 2.0 + mod(row, 2.0) * 0.5);
        float by = fract(bc.y * 2.0);
        float mortar = smoothstep(0.06, 0.0, min(bx, by))
                     + smoothstep(0.06, 0.0, min(1.0 - bx, 1.0 - by));
        soil *= 0.82 + 0.30 * hash21(vec2(floor(bc.x * 2.0), row));
        soil = mix(soil, soil * 0.55, clamp(mortar, 0.0, 1.0));
        if (depth < 0.5) soil = mix(soil, mix(BM.cap, neon, 0.10), 0.85);
        col = soil * (1.05 + 0.75 * exp(-depth * 0.35));
        // Neon lip along the surface, which is where the pack's look lives.
        col += neon * smoothstep(0.10, 0.0, depth)
             * (0.40 + iBass * 1.7 + iPunch * 1.0);
        // The soil itself breathes with the low end.
        col *= 1.0 + iBass * 0.12 * exp(-depth * 0.6);
    }

    // ---- shrubs on the grass line ----
    // Cheap standing decoration: the run used to be bare ground between the
    // features, and bare ground is most of what reads as empty.
    {
        float ti = floor(w.x);
        float gh = groundAt(w.x);
        float hh = hash21(vec2(ti, 3.0));
        if (gh > (-5.0) && hh > 0.80 && w.y > gh - 0.1 && w.y < gh + 0.9) {
            float bw = 0.55 + hash21(vec2(ti, 4.0)) * 0.55;
            vec2 bq = w - vec2(ti + 0.5, gh);
            float d = 1e9;
            for (int k = -1; k <= 1; k++) {
                float fk = float(k);
                d = min(d, length((bq - vec2(fk * 0.30 * bw, 0.0))
                                  / vec2(0.30 * bw, 0.24 + 0.13 * (1.0 - abs(fk)))) - 1.0);
            }
            vec3 bushC = mix(BM.cap * 0.72, palette(iHue + 0.3), 0.10);
            col = mix(col, bushC, smoothstep(0.05, 0.0, d));
            col += neon * smoothstep(0.07, 0.0, abs(d))
                 * (0.22 + iBass * 0.75 + iPunch * 0.5);
        }
    }

    // ---- the flagpole on the biome line ----
    if (iFlag.x > -50.0) {
        float dx = w.x - iFlag.x;
        if (abs(dx) < 0.95) {
            float gh = groundAt(iFlag.x);
            float topY = gh + iFlag.y;
            float pole = box(vec2(dx, w.y - (gh + iFlag.y * 0.5)),
                             vec2(0.045, iFlag.y * 0.5));
            col = mix(col, vec3(0.80, 0.85, 0.90), smoothstep(0.02, 0.0, pole));
            col += neon * smoothstep(0.06, 0.0, abs(pole))
                 * (0.35 + iBeat * 0.9 + iPunch * 1.1);
            float ball = length(w - vec2(iFlag.x, topY + 0.13)) - 0.13;
            col = mix(col, mix(vec3(1.0, 0.86, 0.22), neon, 0.30),
                      smoothstep(0.02, 0.0, ball));
            col += vec3(1.0, 0.86, 0.22) * exp(-max(ball, 0.0) * 7.0)
                 * (0.4 + iPunch * 1.4) * iGlitter;
            // The flag, rippling off the pole.
            vec2 fq = w - vec2(iFlag.x, topY - 0.62);
            float taper = 1.0 - clamp(abs(fq.y) / 0.42, 0.0, 1.0);
            float rip = sin(fq.y * 9.0 + t * 6.0) * 0.05;
            float tri = max(max(-fq.x, fq.x - (0.66 + rip) * taper),
                            abs(fq.y) - 0.42);
            col = mix(col, mix(vec3(0.95, 0.26, 0.24), neon, 0.25),
                      smoothstep(0.02, 0.0, tri));
        }
    }

    // ---- pipes ----
    for (int i = 0; i < 4; i++) {
        vec4 P = iPipe[i];
        if (P.x < -50.0) continue;
        float dx = w.x - P.x;
        if (dx < -0.15 || dx > 1.15) continue;
        float top = groundAt(P.x) + P.y;
        if (w.y < top) {
            bool lip = w.y > top - 0.55;
            float half_ = lip ? 0.62 : 0.50;
            if (abs(dx - 0.5) < half_) {
                vec3 pipeCol = mix(vec3(0.10, 0.74, 0.22), palette(iHue + 0.35), 0.08);
                // A highlight band down the left of the barrel gives it round.
                float shade = 0.55 + 0.65 * exp(-pow((dx - 0.30) * 4.0, 2.0));
                col = pipeCol * shade;
                col += neon * smoothstep(0.05, 0.0, abs(w.y - top)) * 1.1;
                col += neon * smoothstep(0.04, 0.0, abs(abs(dx - 0.5) - half_)) * 0.55;
            }
        }
    }

    // ---- blocks ----
    for (int i = 0; i < 10; i++) {
        vec4 B = iBlock[i];
        if (B.x < -50.0) continue;
        vec2 q = w - vec2(B.x + 0.5, B.y + 0.5 + B.w * 0.25);
        if (abs(q.x) > 0.6 || abs(q.y) > 0.6) continue;
        float d = box(q, vec2(0.46));
        if (d > 0.0) continue;
        if (B.z < 0.5) {
            // Brick: four studs and a mortar cross.
            vec3 c = mix(vec3(0.76, 0.36, 0.16), palette(iHue + 0.6), 0.07);
            float cross_ = smoothstep(0.05, 0.0, min(abs(q.x), abs(q.y)));
            col = c * (0.8 + 0.4 * hash21(vec2(B.x, B.y))) * (1.0 - cross_ * 0.45);
        } else if (B.z < 1.5) {
            // Question block: gold, with a mark built from an arc and a dot.
            vec3 c = mix(vec3(1.0, 0.80, 0.14), palette(iHue + 0.12), 0.07);
            col = c * (0.85 + 0.3 * sin(t * 6.0 + B.x));
            float ring = abs(length(q - vec2(0.0, 0.10)) - 0.15) - 0.045;
            float stem = box(q - vec2(0.055, -0.03), vec2(0.045, 0.10));
            float dot_ = length(q - vec2(0.055, -0.24)) - 0.05;
            float mark = min(max(ring, -q.y + 0.02), min(stem, dot_));
            col = mix(col, vec3(0.25, 0.14, 0.02), smoothstep(0.02, 0.0, mark));
        } else {
            col = mix(vec3(0.42, 0.26, 0.12), palette(iHue + 0.6), 0.18);
        }
        // Each block burns at the band its own column sits in.
        float blb = bandU(clamp((B.x - iCamFrac) / 15.0, 0.0, 1.0));
        col *= 1.0 + blb * 0.30;
        col += neon * smoothstep(0.05, 0.0, abs(d))
             * (0.35 + iBeat * 0.9 + iPunch * 1.1 + blb * 1.3);
    }

    // ---- coins ----
    for (int i = 0; i < 10; i++) {
        vec4 C = iCoin[i];
        if (C.z <= 0.0) continue;
        vec2 q = w - C.xy;
        if (dot(q, q) > 0.36) continue;
        // Spin: the disc narrows to an edge and back.
        float wdt = 0.06 + 0.16 * abs(cos(C.w));
        float d = length(q / vec2(wdt, 0.20)) - 1.0;
        vec3 gold = mix(vec3(1.0, 0.85, 0.20), palette(iHue + 0.1), 0.2);
        col = mix(col, gold, smoothstep(0.08, 0.0, d) * C.z);
        col += gold * exp(-max(d, 0.0) * 5.0) * C.z
             * (0.45 + iTreble * 2.0 + iPunch * 1.2) * iGlitter;
    }

    // ---- enemies ----
    // Three kinds, because one rolling ellipse with eyes on it does not read
    // as a creature -- a goomba is a mushroom, and the silhouette is the whole
    // character: a flared cap, a narrower body and two feet under it.
    for (int i = 0; i < 8; i++) {
        vec4 E = iEnemy[i];
        if (E.x < -50.0) continue;
        vec4 K = iEnemyK[i];
        int kind = int(K.x);
        float squash = clamp(E.z, 0.0, 1.0);
        vec2 q = w - vec2(E.x, E.y);
        if (abs(q.x) > 1.0 || q.y < -0.25 || q.y > 1.9) continue;
        float face = E.w < 0.0 ? -1.0 : 1.0;
        float step_ = sin(K.y);          // walk cycle, from distance covered
        vec3 eNeon = neon;

        if (kind == 2) {
            // ---- piranha plant: a stem out of a pipe, and a mouth ----
            float rise = clamp(K.z, 0.0, 1.0);
            float h = 0.30 + rise * 1.15;
            if (q.y > h + 0.42) continue;
            float stem = box(q - vec2(0.0, h * 0.5), vec2(0.085, h * 0.5));
            vec3 stemC = mix(vec3(0.16, 0.62, 0.18), palette(iHue + 0.35), 0.10);
            col = mix(col, stemC, smoothstep(0.03, 0.0, stem));
            // Leaves, one either side, drooping.
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                vec2 lp = q - vec2(float(sgn) * 0.16, h * 0.42);
                float leaf = length(lp / vec2(0.17, 0.070)) - 1.0;
                col = mix(col, stemC * 1.25, smoothstep(0.10, 0.0, leaf) * rise);
            }
            vec2 hq = q - vec2(0.0, h + 0.26);
            float head = length(hq / vec2(0.34, 0.31)) - 1.0;
            if (head < 0.35) {
                vec3 hc = mix(vec3(0.93, 0.15, 0.15), palette(iHue + 0.02), 0.10);
                col = mix(col, hc, smoothstep(0.08, 0.0, head) * rise);
                // White spots, on a square lattice so they stay round.
                float sp = length(fract(hq * 6.0) - 0.5) - 0.23;
                col = mix(col, vec3(0.97), smoothstep(0.06, 0.0, sp)
                          * smoothstep(0.0, -0.30, head) * rise);
                // The mouth: a wedge opening toward the runner and snapping
                // with the cycle, with a row of teeth along each lip. Without
                // it the plant is a red ball on a stick.
                float snap = 0.30 + 0.70 * (0.5 + 0.5 * step_);
                float gape = (0.10 + rise * 0.22) * snap;
                float inside = max(head + 0.06, -hq.x * face - 0.02);
                float mouth = max(abs(hq.y) - gape - abs(hq.x) * 0.45, inside);
                col = mix(col, vec3(0.34, 0.02, 0.07),
                          smoothstep(0.02, 0.0, mouth) * rise);
                float lip = abs(abs(hq.y) - gape - abs(hq.x) * 0.45) - 0.035;
                float tooth = abs(fract(hq.x * 9.0) - 0.5) - 0.26;
                col = mix(col, vec3(1.0),
                          smoothstep(0.02, 0.0, max(max(lip, tooth), inside)) * rise);
                col += eNeon * smoothstep(0.06, 0.0, abs(head))
                     * (0.35 + iPunch * 0.9 + iBeat * 0.4) * rise;
            }
            continue;
        }

        // Both walkers squash flat when stood on; a koopa keeps its shell.
        // `flat` is a GLSL interpolation qualifier, not a name.
        float squish = kind == 1 ? min(squash, 0.55) : squash;
        float sc = 1.0 - squish * 0.80;
        vec2 r = vec2(q.x, q.y / max(sc, 0.12));

        // Feet: two of them, alternating, which is most of what sells a walk.
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            float ph = step_ * float(sgn);
            vec2 fp = r - vec2(0.17 * float(sgn) + ph * 0.05, 0.075 + max(ph, 0.0) * 0.055);
            float foot = length(fp / vec2(0.155, 0.072)) - 1.0;
            vec3 fc = kind == 1 ? vec3(0.93, 0.74, 0.16) : vec3(0.28, 0.13, 0.05);
            col = mix(col, fc, smoothstep(0.05, 0.0, foot));
        }

        if (kind == 1) {
            // ---- koopa troopa: shell, and a head on a neck ----
            vec3 shell = mix(vec3(0.15, 0.68, 0.22), palette(iHue + 0.40), 0.10);
            vec2 sq = r - vec2(0.0, 0.46);
            float dome = length(sq / vec2(0.46, 0.38)) - 1.0;
            if (squash < 0.5) {
                vec2 hq = r - vec2(0.38 * face, 0.72);
                float head = length(hq / vec2(0.21, 0.185)) - 1.0;
                vec3 skinC = vec3(0.96, 0.80, 0.22);
                col = mix(col, skinC, smoothstep(0.05, 0.0, head));
                float beak = box(hq - vec2(0.14 * face, -0.03), vec2(0.075, 0.035));
                col = mix(col, vec3(0.92, 0.56, 0.10), smoothstep(0.03, 0.0, beak));
                vec2 ep = hq - vec2(0.04 * face, 0.045);
                col = mix(col, vec3(1.0), smoothstep(0.02, 0.0, length(ep / vec2(0.055, 0.070)) - 1.0));
                col = mix(col, vec3(0.05), smoothstep(0.02, 0.0, length(ep - vec2(0.02 * face, 0.0)) - 0.026));
            }
            col = mix(col, shell, smoothstep(0.05, 0.0, dome));
            // Rim and scutes, so the shell is a shell and not a green dome.
            col = mix(col, shell * 0.52, smoothstep(0.04, 0.0, abs(dome + 0.16) - 0.035));
            float sct = abs(fract(atan(sq.y, sq.x) * 1.9) - 0.5) - 0.30;
            col = mix(col, shell * 0.62,
                      smoothstep(0.04, 0.0, max(sct, dome + 0.20)));
            col += eNeon * smoothstep(0.06, 0.0, abs(dome))
                 * (0.35 + iPunch * 0.8 + iBeat * 0.35);
        } else {
            // ---- goomba: a mushroom, cap flared over a narrower body ----
            vec3 capC = mix(vec3(0.62, 0.33, 0.13), palette(iHue + 0.50), 0.08);
            vec2 cq = r - vec2(0.0, 0.52);
            float cap = length(cq / vec2(0.52, 0.34)) - 1.0;
            float bodyB = box(r - vec2(0.0, 0.30), vec2(0.30, 0.23)) - 0.07;
            float body = min(cap, bodyB);
            col = mix(col, capC, smoothstep(0.05, 0.0, body));
            // The lighter face the prints always gave it, on the cap's front.
            float faceP = length((r - vec2(0.0, 0.40)) / vec2(0.35, 0.23)) - 1.0;
            col = mix(col, mix(capC, vec3(0.94, 0.80, 0.58), 0.72),
                      smoothstep(0.05, 0.0, faceP));
            if (squash < 0.5) {
                for (int sgn = -1; sgn <= 1; sgn += 2) {
                    vec2 ep = r - vec2(0.150 * float(sgn), 0.47);
                    col = mix(col, vec3(1.0),
                              smoothstep(0.02, 0.0, length(ep / vec2(0.075, 0.105)) - 1.0));
                    col = mix(col, vec3(0.05),
                              smoothstep(0.02, 0.0, length(ep - vec2(0.028 * face, 0.0)) - 0.033));
                    // The scowl: a bar angled down toward the middle.
                    vec2 bq = r - vec2(0.150 * float(sgn), 0.595);
                    float a = 0.42 * float(sgn);
                    bq = mat2(cos(a), -sin(a), sin(a), cos(a)) * bq;
                    col = mix(col, vec3(0.15, 0.06, 0.02),
                              smoothstep(0.02, 0.0, box(bq, vec2(0.10, 0.026))));
                }
            }
            col += eNeon * smoothstep(0.06, 0.0, abs(body))
                 * (0.35 + iPunch * 0.8 + iBeat * 0.35);
        }
    }

    // ---- event sparks ----
    // Thrown by things that actually happen -- a stomp, a block opening, a
    // coin, a landing -- rather than emitted on a timer, so the sparkle lands
    // on the same frames the music does.
    for (int i = 0; i < 28; i++) {
        vec4 S = iPart[i];
        if (S.z <= 0.0) continue;
        vec2 q = w - S.xy;
        float r2 = dot(q, q);
        if (r2 > 0.16) continue;
        vec3 pc = S.w < 0.5 ? vec3(0.95, 0.92, 0.85)
                : (S.w < 1.5 ? vec3(1.0, 0.86, 0.25)
                             : mix(vec3(1.0, 0.55, 0.20), neon, 0.35));
        float sz = 0.016 + 0.030 * S.z;
        col += pc * exp(-r2 / (sz * sz) * 0.5)
             * S.z * (0.8 + iTreble * 0.9 + iPunch * 0.8) * iGlitter;
    }

    // ---- the runner ----
    {
        vec2 q = w - vec2(iHero.x, iHero.y);
        if (abs(q.x) < 0.9 && q.y > -0.2 && q.y < 1.4) {
            float f = iHero.z;
            vec3 shirt = mix(vec3(0.90, 0.14, 0.12), palette(iHue + 0.05), 0.15);
            vec3 jeans = mix(vec3(0.16, 0.30, 0.85), palette(iHue + 0.45), 0.15);
            vec3 skin = vec3(1.0, 0.80, 0.62);
            // Legs stride while grounded, tuck while airborne.
            float stride = sin(t * 16.0 + iHero.x * 3.0) * (1.0 - iHeroAir);
            float cover = 0.0;
            float legs = min(box(q - vec2(-0.10 + stride * 0.09, 0.16), vec2(0.09, 0.17)),
                             box(q - vec2(0.10 - stride * 0.09, 0.16), vec2(0.09, 0.17)));
            float m = smoothstep(0.03, 0.0, legs);
            col = mix(col, jeans, m); cover = max(cover, m);
            float torso = box(q - vec2(0.0, 0.52), vec2(0.20, 0.22));
            m = smoothstep(0.03, 0.0, torso);
            col = mix(col, shirt, m); cover = max(cover, m);
            float head = length(q - vec2(0.02 * f, 0.86)) - 0.19;
            m = smoothstep(0.03, 0.0, head);
            col = mix(col, skin, m); cover = max(cover, m);
            // Cap, brim forward, and the moustache.
            float cap = min(box(q - vec2(0.02 * f, 0.99), vec2(0.19, 0.08)),
                            box(q - vec2(0.16 * f, 0.95), vec2(0.12, 0.04)));
            m = smoothstep(0.03, 0.0, cap);
            col = mix(col, shirt, m); cover = max(cover, m);
            float tash = box(q - vec2(0.09 * f, 0.80), vec2(0.10, 0.035));
            col = mix(col, vec3(0.18, 0.09, 0.04), smoothstep(0.02, 0.0, tash));
            // An arm thrown up on the way up sells the jump.
            float arm = box(q - vec2(0.24 * f, 0.62 + iHeroAir * 0.22), vec2(0.07, 0.16));
            m = smoothstep(0.03, 0.0, arm);
            col = mix(col, shirt, m); cover = max(cover, m);
            col += palette(iHue + 0.5) * iHeroStumble * cover * 0.8;
        }
    }

    // Sparkle, and the beat on the whole frame.
    vec2 sp = w * 6.0;
    vec2 sg = floor(sp) + floor(t * 12.0);
    if (hash21(sg) > 0.9965) {
        vec2 sd = fract(sp) - vec2(hash21(sg + 3.1), hash21(sg + 7.7));
        col += vec3(1.0) * exp(-dot(sd, sd) * 26.0)
             * (0.28 + iTreble * 1.6 + iPunch * 0.9) * iGlitter;
    }
    col += neon * (iBeat * 0.07 + iPunch * 0.10);
    // Aspect-corrected, or the vignette is an ellipse on a wide screen.
    // At 1.05/0.30 this put the corners at 0.3% brightness and the side
    // edges at 11%, which on the darker biomes read as the frame being murky
    // rather than as a vignette.
    col *= smoothstep(1.50, 0.58, length((v_uv - 0.5) * vec2(A, 1.0)));
    col = col / (1.0 + col * 0.30);
    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.85)), 1.0);
}
"""


class SuperMario(BaseEffect):
    """Auto-running platformer: real terrain, real gravity, real decisions."""

    NAME = 'Super Mario'
    AUTHOR = 'unicorn-viz'
    TAGS = ['games', 'arcade', 'neon', 'energetic', 'groovy', 'hard']
    PING_PONG_FRIENDS = [
        'Donkey Kong', 'Mario Kart', 'Neon Pac', 'Galaga', 'Breakout',
        'Audio Spectrum',
    ]

    # 7.0 with a 0.8-1.35 tempo band ran him at up to 9.5 tiles/s, which is
    # most of the visible width every two seconds -- too fast to read the level
    # he is running through.
    _RUN = 4.9              # tiles / s at rest
    _JUMP_V = 21.0          # tiles / s launch
    _CAM_LEAD = 4.5         # tiles the runner sits behind the camera's left edge

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
        # ever moves at its left-hand end and the skyline looks half dead.
        self._band_tilt = np.linspace(1.0, 2.7, 16)
        self._tbl_band = np.zeros((4, 4), dtype='f4')
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._sky = float(self.rng.uniform(0.0, 1.0))
        self._seed = float(self.rng.uniform(0.0, 1000.0))

        # Level state. Columns are generated once, on demand, and cached by
        # absolute tile index so the world behind the runner is the world he
        # already ran through.
        self._col_cache: dict[int, float] = {}
        self._feature_cache: dict[int, None] = {}
        self._blocks: list[dict] = []
        self._enemies: list[dict] = []
        self._pipes: list[dict] = []
        self._coins = np.zeros((_NCOIN, 5), dtype=np.float64)   # x y vy life spin
        # x, y, vx, vy, life, kind
        self._parts = np.zeros((_NPART, 6), dtype=np.float64)
        self._next_part = 0

        self._cam = float(self.rng.uniform(0.0, 500.0))
        self._hero_x = self._cam + self._CAM_LEAD
        self._hero_y = float(_GROUND_Y)
        self._hero_vy = 0.0
        self._air = 0.0
        self._coyote = 0.0
        self._stumble = 0.0
        self._cam_y = 0.0
        self._view_h = 11.25
        self._tbl_flag = (-99.0, 0.0)

        self._tbl_ground = np.zeros(_COLS, dtype='f4')
        self._tbl_block = np.zeros((_NBLOCK, 4), dtype='f4')
        self._tbl_enemy = np.zeros((_NENEMY, 4), dtype='f4')
        self._tbl_enemyk = np.zeros((_NENEMY, 4), dtype='f4')
        self._tbl_coin = np.zeros((_NCOIN, 4), dtype='f4')
        self._tbl_part = np.zeros((_NPART, 4), dtype='f4')
        self._tbl_pipe = np.zeros((4, 4), dtype='f4')
        # Run the level in a little so the first frame is mid-level, never a
        # blank start on an empty plain.
        for _ in range(240):
            self._step(1.0 / 60.0, 1.0)

    # ---------------------------------------------------------------- level
    def _ground(self, tile: int) -> float:
        """Ground top at an absolute tile index, generated once and cached."""
        got = self._col_cache.get(tile)
        if got is not None:
            return got
        h = _GROUND_Y
        # A long, slow terrace so the run is not dead flat.
        h += math.floor(1.6 + 1.6 * math.sin(tile * 0.045 + self._seed * 0.01)) - 1.0
        # Pits are decided per nine-tile block, not per tile: deciding per tile
        # meant the two conditions could only ever agree on a single column, so
        # a "gap" was one missing tile and the runner barely had to jump.
        # Capped at three wide because that is what a jump actually clears --
        # airtime is 2*v/g = 0.68s, which carries him 4.8 tiles at base speed.
        blk = tile // 9
        if self._rand(blk, 11) > 0.62:
            span = 2 + int(self._rand(blk, 12) * 2.0)
            start = 3 + int(self._rand(blk, 13) * 3.0)
            off = tile - blk * 9
            if start <= off < start + span:
                h = _PIT
        self._col_cache[tile] = h
        if len(self._col_cache) > 4000:
            cutoff = int(self._cam) - 80
            for k in [k for k in self._col_cache if k < cutoff]:
                del self._col_cache[k]
        return h

    def _burst(self, x: float, y: float, n: int, kind: float,
               spread: float = 5.0) -> None:
        """Throw *n* sparks from a point. Ring-allocated, so a busy frame
        overwrites the oldest rather than dropping the newest."""
        for _ in range(n):
            i = self._next_part
            self._next_part = (self._next_part + 1) % _NPART
            a = float(self.rng.uniform(0.0, 2.0 * math.pi))
            sp = float(self.rng.uniform(0.35, 1.0)) * spread
            self._parts[i] = (x, y, math.cos(a) * sp, abs(math.sin(a)) * sp,
                              1.0, kind)

    def _step_parts(self, dt: float) -> None:
        """Integrate the sparks; they fall, slow down and fade."""
        live = self._parts[:, 4] > 0.0
        if not live.any():
            return
        self._parts[live, 0] += self._parts[live, 2] * dt
        self._parts[live, 1] += self._parts[live, 3] * dt
        self._parts[live, 3] -= _GRAVITY * 0.30 * dt
        drag = math.exp(-2.4 * dt)
        self._parts[live, 2] *= drag
        self._parts[live, 4] -= dt * 1.5
        np.maximum(self._parts[:, 4], 0.0, out=self._parts[:, 4])

    def _rand(self, tile: int, salt: int) -> float:
        """Deterministic per-tile noise, so a column is the same every time."""
        x = math.sin(tile * 12.9898 + salt * 78.233 + self._seed) * 43758.5453
        return x - math.floor(x)

    def _spawn_features(self, tile: int) -> None:
        """Place blocks, pipes and goombas as a column comes into existence."""
        if tile in self._feature_cache:
            return
        self._feature_cache[tile] = None
        if len(self._feature_cache) > 4000:
            cutoff = int(self._cam) - 80
            for k in [k for k in self._feature_cache if k < cutoff]:
                del self._feature_cache[k]
        g = self._ground(tile)
        if g <= _PIT:
            return
        r = self._rand(tile, 3)
        far_enough = all(abs(q['x'] - tile) > 7.0 for q in self._pipes)
        if r > 0.93 and far_enough and len(self._pipes) < 4:
            pipe = {'x': float(tile), 'h': 1.0 + math.floor(self._rand(tile, 4) * 2.0)}
            self._pipes.append(pipe)
            # Better than half of them have something living in them.
            if self._rand(tile, 8) > 0.42 and len(self._enemies) < _NENEMY:
                self._enemies.append({
                    'x': float(tile) + 0.5, 'y': g + pipe['h'], 'squash': 0.0,
                    'face': -1.0, 'kind': 2.0,
                    'walk': float(self._rand(tile, 9) * 6.28),
                    'rise': 0.0, 'pipe': pipe})
        elif r > 0.80 and len(self._blocks) < _NBLOCK:
            kind = 1.0 if self._rand(tile, 5) > 0.45 else 0.0
            self._blocks.append({'x': float(tile), 'y': g + 3.5, 'kind': kind, 'hit': 0.0})
        if self._rand(tile, 6) > 0.88 and len(self._enemies) < _NENEMY:
            # Koopas are the rarer of the two walkers, and slower on their feet.
            kind = 1.0 if self._rand(tile, 7) > 0.62 else 0.0
            self._enemies.append({'x': float(tile), 'y': g, 'squash': 0.0,
                                  'face': -1.0, 'kind': kind, 'walk': 0.0,
                                  'rise': 0.0, 'pipe': None})

    # ----------------------------------------------------------------- sim
    def _gap_ahead(self, x: float) -> bool:
        """True when the ground runs out within jumping distance."""
        for d in (1.0, 1.6, 2.2):
            if self._ground(int(math.floor(x + d))) <= _PIT:
                return True
        return False

    def _enemy_ahead(self, x: float) -> bool:
        return any(0.4 < e['x'] - x < 2.0 and e['squash'] < 0.5 for e in self._enemies)

    def _block_ahead(self, x: float, y: float, run: float) -> bool:
        """True when jumping *now* would put his head under a question block.

        A fixed distance window cannot express this. How far ahead the block
        should be when he leaves the ground is a physics quantity: his head has
        to rise `h` to reach the block's underside, which takes
        `t = (v - sqrt(v^2 - 2gh)) / g`, and in that time he covers `run * t`.
        Both terms move -- `v` with the bass, `run` with the tempo -- so the
        right lead is between half a tile and two tiles depending on the track,
        and a window fixed at 0.8-2.4 catches it only at one end. Measured over
        three minutes with the fixed window: 7 blocks opened.
        """
        v = self._jump_v()
        for b in self._blocks:
            if b['kind'] != 1.0:
                continue
            h = b['y'] - (y + _HEAD)
            if h <= 0.15 or 2.0 * _GRAVITY * h >= v * v:
                continue
            t = (v - math.sqrt(v * v - 2.0 * _GRAVITY * h)) / _GRAVITY
            if abs(b['x'] + 0.5 - x - run * t) < 0.40:
                return True
        return False

    def _jump_v(self) -> float:
        """Launch speed for a jump taken this frame."""
        return self._JUMP_V * (0.85 + self._bass * 0.30)

    def _step(self, dt: float, tempo: float) -> None:
        """One tick of the level, the runner and everything he interacts with."""
        run = self._RUN * tempo
        self._hero_x += run * dt

        grounded = self._air < 0.5
        gnd = self._ground(int(math.floor(self._hero_x)))
        # Coyote time: a jump asked for just after leaving an edge still fires,
        # which is the difference between a runner and a physics toy.
        self._coyote = 0.08 if grounded else max(0.0, self._coyote - dt)

        want = (self._gap_ahead(self._hero_x)
                or self._enemy_ahead(self._hero_x)
                or self._block_ahead(self._hero_x, self._hero_y, run))
        if want and self._coyote > 0.0 and self._hero_vy <= 0.0:
            self._hero_vy = self._jump_v()
            self._coyote = 0.0

        self._hero_vy -= _GRAVITY * dt
        prev_y = self._hero_y
        self._hero_y += self._hero_vy * dt

        if gnd > _PIT and self._hero_y <= gnd and self._hero_vy <= 0.0:
            if self._air > 0.5 and self._hero_vy < -8.0:
                # Dust on touchdown, scaled by how hard he came down.
                self._burst(self._hero_x, gnd + 0.05,
                            min(7, 2 + int(-self._hero_vy / 5.0)), 0.0, 3.4)
            self._hero_y = gnd
            self._hero_vy = 0.0
            self._air = 0.0
        else:
            self._air = 1.0
        if self._hero_y < -3.0:                 # fell in a pit: put him back
            self._hero_y = max(_GROUND_Y, gnd if gnd > _PIT else _GROUND_Y) + 6.0
            self._hero_vy = 0.0
            self._stumble = 1.0

        self._cam = self._hero_x - self._CAM_LEAD
        # Lift the camera only once he climbs past the upper third, and let it
        # down gently. Without it a jump from a high terrace apexes off the top
        # of the frame -- v^2/2g is 3.6 tiles of rise on top of the terrace.
        # Only once he is genuinely near the top of the frame. At 0.62 it
        # triggered on nearly every jump, never settled between them, and the
        # ground spent most of the run pushed off the bottom edge.
        want_y = max(0.0, self._hero_y - self._view_h * 0.82)
        rate = 8.0 if want_y > self._cam_y else 3.0
        self._cam_y += (want_y - self._cam_y) * min(1.0, dt * rate)
        for tile in range(int(self._cam), int(self._cam) + _COLS + 2):
            self._spawn_features(tile)

        # Blocks: opened from below, and they pop a coin when they are.
        for b in self._blocks:
            b['hit'] = max(0.0, b['hit'] - dt * 6.0)
            # Swept against the block's underside. A fixed window cannot work
            # here: he rises 21 tiles/s, which is 0.35 tiles per frame, so any
            # window narrower than a frame's travel is a one-frame target and
            # is usually stepped straight over.
            head_now = self._hero_y + _HEAD
            head_prev = prev_y + _HEAD
            if (b['kind'] == 1.0 and abs(b['x'] + 0.5 - self._hero_x) < 0.75
                    and self._hero_vy > 0.0
                    and head_prev < b['y'] <= head_now):
                b['kind'] = 2.0
                b['hit'] = 1.0
                self._pop_coin(b['x'] + 0.5, b['y'] + 1.0)
                self._burst(b['x'] + 0.5, b['y'] + 1.0, 8, 1.0, 4.0)
        self._blocks = [b for b in self._blocks if b['x'] > self._cam - 6.0]
        self._pipes = [q for q in self._pipes if q['x'] > self._cam - 6.0]

        # The walkers walk and get stood on; the plants come and go.
        for e in self._enemies:
            kind = e['kind']
            if kind >= 2.0:
                # A piranha plant on a cycle -- but it will not come up while
                # the runner is standing over its pipe, which is the rule the
                # originals used and the reason the pipes stay crossable.
                e['walk'] += dt * 1.35 * tempo
                over = abs(self._hero_x - e['x']) < 1.15
                want = 0.0 if over else max(0.0, math.sin(e['walk']))
                e['rise'] += (want - e['rise']) * min(1.0, dt * 3.2)
                if (e['rise'] > 0.35 and abs(self._hero_x - e['x']) < 0.55
                        and self._hero_y < e['y'] + 1.5):
                    self._stumble = 1.0
                continue
            if e['squash'] >= 1.0:
                continue
            if e['squash'] > 0.0:
                # A stomped koopa stops at its shell and stays on the board;
                # a goomba goes flat and is cleared.
                cap = 0.55 if kind >= 1.0 else 1.0
                e['squash'] = min(cap, e['squash'] + dt * 3.0)
                if kind >= 1.0:
                    continue
                continue
            speed = 1.55 if kind >= 1.0 else 1.95
            step = e['face'] * speed * dt * tempo
            e['x'] += step
            e['walk'] += abs(step) * 6.5
            eg = self._ground(int(math.floor(e['x'])))
            if eg <= _PIT:
                e['x'] = -999.0
                continue
            e['y'] = eg
            dx = self._hero_x - e['x']
            if abs(dx) < 0.45 and abs(self._hero_y - e['y']) < 1.0:
                if self._hero_vy < 0.0 and self._hero_y > e['y'] + 0.3:
                    e['squash'] = 0.01
                    self._hero_vy = self._JUMP_V * 0.55     # the stomp bounce
                    self._burst(e['x'], e['y'] + 0.45, 10, 1.0)
                else:
                    self._stumble = 1.0
        self._enemies = [e for e in self._enemies
                         if e['x'] > self._cam - 6.0 and e['squash'] < 1.0
                         and not (e['kind'] >= 2.0 and e['x'] < self._cam - 4.0)]

        live = self._coins[:, 3] > 0.0
        if live.any():
            self._coins[live, 1] += self._coins[live, 2] * dt
            self._coins[live, 2] -= _GRAVITY * 0.45 * dt
            self._coins[live, 3] -= dt * 0.9
            self._coins[live, 4] += dt * 9.0
        self._stumble = max(0.0, self._stumble - dt * 1.8)
        self._step_parts(dt)

    def _pop_coin(self, x: float, y: float) -> None:
        """Throw a coin up out of an opened block."""
        free = np.where(self._coins[:, 3] <= 0.0)[0]
        if not free.size:
            return
        self._coins[int(free[0])] = (x, y, 13.0, 1.0, float(self.rng.uniform(0.0, 6.0)))

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
        """Advance the run; the beat buys a jump when one is available."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 1.8))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 1.8))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 1.8))
        self._pump_bands(audio, dt)
        if audio.beat > 0.5:
            self._beat = 1.0
            self._punch = 1.0
            # Not every beat: jumping on all of them kept him in the air and
            # the run cycle never showed.
            if (self._coyote > 0.0 and self._hero_vy <= 0.0
                    and self.rng.random() < 0.28):
                self._hero_vy = self._JUMP_V * (0.9 + self._bass * 0.45)
                self._coyote = 0.0
            # A glint off his feet on every beat he is grounded for, so the run
            # reads as on the beat even between jumps.
            if self._air < 0.5:
                self._burst(self._hero_x, self._hero_y + 0.06, 3, 0.0, 2.6)
        self._beat = max(0.0, self._beat - dt * 3.4)
        # A second, much faster envelope. `iBeat` at 3.4/s is a ~300 ms swell,
        # which reads as a pulse rather than as a hit; the transient needs
        # something that is gone again before the next one arrives.
        self._punch = max(0.0, self._punch - dt * 11.0)

        tempo = max(0.05, float(self.parameters['speed'])) * (0.86 + self._bass * 0.34)
        self._step(min(dt, 1.0 / 30.0), tempo)

    def render(self) -> None:
        """Upload the level and everything standing on it."""
        base = int(math.floor(self._cam))
        for i in range(_COLS):
            self._tbl_ground[i] = self._ground(base + i)

        self._tbl_block[:] = _OFF
        for i, b in enumerate(self._blocks[:_NBLOCK]):
            self._tbl_block[i] = (b['x'] - base, b['y'], b['kind'], b['hit'])
        self._tbl_part[:] = 0.0
        for i in range(_NPART):
            if self._parts[i, 4] > 0.0:
                self._tbl_part[i] = (self._parts[i, 0] - base, self._parts[i, 1],
                                     self._parts[i, 4], self._parts[i, 5])
        self._tbl_enemy[:] = _OFF
        for i, e in enumerate(self._enemies[:_NENEMY]):
            self._tbl_enemy[i] = (e['x'] - base, e['y'], e['squash'], e['face'])
            self._tbl_enemyk[i] = (e['kind'], e['walk'], e['rise'], 0.0)
        self._tbl_pipe[:] = _OFF
        for i, q in enumerate(self._pipes[:4]):
            self._tbl_pipe[i] = (q['x'] - base, q['h'], 0.0, 0.0)
        self._tbl_coin[:, 0] = self._coins[:, 0] - base
        self._tbl_coin[:, 1] = self._coins[:, 1]
        self._tbl_coin[:, 2] = np.clip(self._coins[:, 3], 0.0, 1.0)
        self._tbl_coin[:, 3] = self._coins[:, 4]

        p = self._prog
        p['iResolution'].value = (float(self.width), float(self.height))
        self._view_h = _VIEW_W / max(float(self.width) / max(float(self.height), 1.0), 0.1)
        p['iTime'].value = float(self.time)
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iPunch'].value = self._punch
        p['iBand'].write(self._tbl_band.tobytes())
        p['iHue'].value = self._hue
        p['iGlitter'].value = max(0.0, float(self.parameters['glitter']))
        p['iSky'].value = self._sky
        # One biome per ninety tiles, crossfading over the last nine so the
        # change is a place you run into rather than a cut.
        zf = self._cam / 90.0
        p['iZone'].value = float(math.floor(zf) + _smoothstep(0.90, 1.0, zf % 1.0))
        # The flagpole stands on the boundary tile, if it is in view.
        line = (math.floor(zf) + 1.0) * 90.0
        if -2.0 < line - self._cam < _VIEW_W + 2.0 and self._ground(int(line)) > _PIT:
            p['iFlag'].value = (float(line - base), 4.6)
        else:
            p['iFlag'].value = (-99.0, 0.0)
        p['iCamFrac'].value = float(self._cam - base)
        p['iCamY'].value = float(self._cam_y)
        p['iViewH'].value = float(self._view_h)
        p['iGround'].write(self._tbl_ground.tobytes())
        p['iBlock'].write(self._tbl_block.tobytes())
        p['iEnemy'].write(self._tbl_enemy.tobytes())
        p['iEnemyK'].write(self._tbl_enemyk.tobytes())
        p['iCoin'].write(self._tbl_coin.tobytes())
        p['iPart'].write(self._tbl_part.tobytes())
        p['iPipe'].write(self._tbl_pipe.tobytes())
        p['iHero'].value = (float(self._hero_x - base), float(self._hero_y),
                            1.0)
        p['iHeroAir'].value = float(self._air)
        p['iHeroStumble'].value = float(self._stumble)
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()

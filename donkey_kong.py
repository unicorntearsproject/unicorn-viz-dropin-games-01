"""Donkey Kong — a real girder-climbing platformer simulation rendered as a
neon light-show.

Same architecture as the rest of the pack: an actual climb runs on the CPU
(five zig-zag girder platforms, ladders, a self-playing hero who paths toward
the next ladder up and hops over rolling barrels, real barrel physics rolling
downhill, dropping a level at each girder's low end, and sometimes taking a
ladder down) and the shader renders the true state as glowing neon.

The cast is drawn as real sprites rather than blobs: a small overalls-and-cap
hero with running legs and swinging arms, a big ape up top who beats his chest
on the beat and winds up a throwing arm for every barrel, and a damsel on the
far end of the top girder. Reach the top and the rig flashes, hearts rise
between the pair, and the climb restarts — perpetual, so the show never ends.

A hammer power-up periodically appears on a girder. Grab it and the hero goes
hunting: barrels struck while it is up burst into sparks instead of killing.
Classic rule kept — you cannot climb while carrying it, so taking the hammer
is a real trade-off the AI actually makes.

Audio reactivity:
  - bass    -> barrel roll speed, girder pulse, barrel glow
  - mid     -> hero climb speed, hero glow, palette drift
  - treble  -> spark/rivet sparkle on girders, smash-burst sparkle
  - beat    -> screen flash, palette shift, Kong chest-beat

Randomised per activation: palette hue, hero starting phase, barrel-throw and
hammer timers, and ``self.time`` — no two runs look or play alike from frame 0.
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

_NROWS = 5
_NBARREL = 8
_NLADDER = (_NROWS - 1) * 2
_NSMASH = 10
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
uniform float iWin;           // celebration envelope [0,1]
uniform vec4  iRow[5];        // (x0, y0, x1, y1) girder endpoints, world space
uniform vec4  iLadder[8];     // (x, yBottom, yTop, _)
uniform int   iLadderCount;
uniform vec2  iHeroPos;       // hero FEET position, world space
uniform float iHeroFace;
uniform float iHeroClimb;     // 0 walking, 1 climbing
uniform float iHeroHit;       // death-flash blink
uniform float iHeroHammer;    // >0 while the hammer is held
uniform vec2  iKongPos;       // Kong FEET position
uniform float iKongThrow;     // 0..1 throw-arm swing envelope
uniform float iKongRage;      // 0..1 chest-beat envelope
uniform float iKongPound;     // chest-drum phase, radians, CPU-integrated
uniform vec2  iDamselPos;     // damsel FEET position
uniform vec2  iOilPos;        // oil-drum base
uniform vec2  iBarrelPos[8];
uniform float iBarrelRot[8];
uniform float iBarrelAlive[8];
uniform float iBarrelFire[8]; // 1 = flaming barrel variant
uniform vec2  iHammerPos;
uniform float iHammerAlive;
uniform vec3  iSmash[10];     // (x, y, age); age<0 inactive
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
mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

float segDist(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}
// Soft filled ellipse mask centred at c with radii r.
float ell(vec2 q, vec2 c, vec2 r, float soft) {
    return smoothstep(soft, -soft, length((q - c) / r) - 1.0);
}

vec3 backdrop(vec2 uv, float A, float t) {
    vec2 p = (uv - 0.5) * vec2(A, 1.0);
    float v = sin(p.x * 2.1 + t * 0.3) + sin(p.y * 1.8 - t * 0.24);
    vec3 col = palette(v * 0.08 + iHue + 0.6 + t * 0.014);
    col *= col;
    col *= 0.05 + 0.05 * (v * 0.5 + 0.5);
    // Stars: a soft round point placed inside its cell. Filling the whole cell
    // instead -- a bare threshold on the cell hash -- is what renders as a
    // field of hard little squares.
    vec2 scell = p * 100.0;
    vec2 sg = floor(scell);
    float sh = hash21(sg);
    if (sh > 0.972) {
        vec2 sd = fract(scell) - vec2(hash21(sg + 5.3), hash21(sg + 9.1));
        float tw = 0.5 + 0.5 * sin(t * (2.0 + sh * 6.0) + sh * 40.0);
        col += vec3(0.6, 0.7, 0.9) * exp(-dot(sd, sd) * 26.0)
             * tw * (0.35 + iTreble * 0.45);
    }
    return col;
}

vec3 drawGirder(vec2 p, vec4 row, vec3 col) {
    vec2 a = row.xy, b = row.zw;
    float d = segDist(p, a, b);
    if (d > 0.05) return vec3(0.0);
    float body = smoothstep(0.016, 0.009, d);
    float rivetPhase = fract(dot(p - a, normalize(b - a)) * 14.0);
    float rivet = smoothstep(0.06, 0.0, min(rivetPhase, 1.0 - rivetPhase)) * body;
    vec3 o = col * body * 0.7;
    o += col * rivet * (0.5 + iTreble * 0.5);
    o += col * smoothstep(0.020, 0.0, d) * 0.4 * (1.0 + iBass * 0.5);
    return o;
}

vec3 drawLadder(vec2 p, vec4 lad, vec3 col) {
    float x = lad.x, y0 = lad.y, y1 = lad.z;
    if (abs(p.x - x) > 0.030 || p.y < y0 - 0.01 || p.y > y1 + 0.01) return vec3(0.0);
    float rail = smoothstep(0.0035, 0.0, abs(abs(p.x - x) - 0.016));
    // Distance (in world units) to the nearest rung line, every 0.024.
    float rungGap = 0.024;
    float ry = abs(fract((p.y - y0) / rungGap) - 0.5) * rungGap;
    float rung = smoothstep(0.0035, 0.0, ry) * step(abs(p.x - x), 0.017);
    return col * clamp(rail + rung, 0.0, 1.0) * 0.85;
}

// Hero: overalls-and-cap climber, FEET at q=(0,0), ~0.10 tall, half-width 0.024.
vec3 drawHero(vec2 q, float face, float climb, float hammer, float t) {
    if (abs(q.x) > 0.075 || q.y < -0.01 || q.y > 0.135) return vec3(0.0);
    q.x *= face;
    vec3 skin = vec3(1.00, 0.80, 0.60);
    vec3 shirt = vec3(1.00, 0.16, 0.18);
    vec3 denim = vec3(0.20, 0.36, 1.00);
    vec3 col = vec3(0.0);

    float gait = sin(t * (climb > 0.5 ? 11.0 : 17.0));
    float spread = climb > 0.5 ? 0.009 : gait * 0.013;
    float legs = smoothstep(0.008, 0.0, segDist(q, vec2(-0.006, 0.032), vec2(-0.006 - spread, 0.0)))
               + smoothstep(0.008, 0.0, segDist(q, vec2( 0.006, 0.032), vec2( 0.006 + spread, 0.0)));
    col = mix(col, denim, clamp(legs, 0.0, 1.0));
    // Boots.
    float boots = ell(q, vec2(-0.006 - spread, 0.004), vec2(0.011, 0.006), 0.004)
                + ell(q, vec2( 0.006 + spread, 0.004), vec2(0.011, 0.006), 0.004);
    col = mix(col, vec3(0.35, 0.16, 0.06), clamp(boots, 0.0, 1.0));

    col = mix(col, denim, ell(q, vec2(0.0, 0.046), vec2(0.019, 0.021), 0.005));
    col = mix(col, shirt, ell(q, vec2(0.0, 0.060), vec2(0.020, 0.011), 0.005));
    // Overall straps read across the shirt.
    col = mix(col, denim, smoothstep(0.004, 0.0, abs(abs(q.x) - 0.009))
                          * step(0.050, q.y) * step(q.y, 0.066));

    float swing = climb > 0.5 ? abs(gait) * 0.014 : gait * 0.016;
    float arm = smoothstep(0.008, 0.0, segDist(q, vec2(0.012, 0.060), vec2(0.024, 0.046 + swing)))
              + smoothstep(0.008, 0.0, segDist(q, vec2(-0.012, 0.060), vec2(-0.024, 0.046 - swing)));
    col = mix(col, shirt, clamp(arm, 0.0, 1.0));
    col = mix(col, skin, ell(q, vec2(0.024, 0.046 + swing), vec2(0.008, 0.008), 0.004));

    col = mix(col, skin, ell(q, vec2(0.0, 0.079), vec2(0.017, 0.016), 0.005));
    col = mix(col, vec3(0.25, 0.10, 0.03), ell(q, vec2(0.005, 0.073), vec2(0.012, 0.004), 0.003));
    col = mix(col, vec3(0.05, 0.03, 0.02), ell(q, vec2(0.008, 0.082), vec2(0.003, 0.004), 0.002));
    // Cap: crown plus a forward brim.
    col = mix(col, shirt, ell(q, vec2(0.0, 0.093), vec2(0.019, 0.010), 0.004));
    col = mix(col, shirt * 0.85, ell(q, vec2(0.013, 0.090), vec2(0.015, 0.004), 0.003));

    if (hammer > 0.0) {
        // Overhead swing: the hammer arcs forward and slams down.
        float sw = sin(t * 13.0);
        vec2 hq = rot(-0.9 + sw * 0.9) * (q - vec2(0.004, 0.070));
        float shaft = smoothstep(0.005, 0.0, segDist(hq, vec2(0.0, 0.0), vec2(0.0, 0.052)));
        col = mix(col, vec3(0.55, 0.30, 0.12), shaft);
        float headM = ell(hq, vec2(0.0, 0.058), vec2(0.020, 0.013), 0.005);
        col = mix(col, vec3(1.0, 0.85, 0.25), headM);
        col += vec3(1.0, 0.7, 0.2) * headM * (0.5 + 0.5 * abs(sw));
    }
    return col;
}

// One jointed arm: shoulder -> elbow -> fist, as two segments plus a knuckle.
// Angles are measured CCW from +x, so a pose is stated as where the upper arm
// and the forearm each point, and the elbow flexes between them.
float kongArm(vec2 q, vec2 sh, float t0, float t1, float r) {
    vec2 el = sh + 0.030 * vec2(cos(t0), sin(t0));
    vec2 fi = el + 0.028 * vec2(cos(t1), sin(t1));
    float d = min(segDist(q, sh, el), segDist(q, el, fi));
    return clamp(smoothstep(r, r * 0.40, d)
               + smoothstep(r * 1.35, r * 0.50, length(q - fi)), 0.0, 1.0);
}

// Kong: FEET at q=(0,0), ~0.175 tall, half-width ~0.095.
vec3 drawKong(vec2 q, float throwT, float rage) {
    if (abs(q.x) > 0.16 || q.y < -0.01 || q.y > 0.22) return vec3(0.0);
    vec3 fur = vec3(0.50, 0.26, 0.10);
    vec3 pale = vec3(0.82, 0.62, 0.42);
    vec3 col = vec3(0.0);

    float legs = ell(q, vec2(-0.038, 0.024), vec2(0.028, 0.026), 0.007)
               + ell(q, vec2( 0.038, 0.024), vec2(0.028, 0.026), 0.007);
    col = mix(col, fur * 0.85, clamp(legs, 0.0, 1.0));
    col = mix(col, fur, ell(q, vec2(0.0, 0.082), vec2(0.072, 0.055), 0.010));
    // Chest patch swells on the beat — the chest-beat tell.
    float beatPuff = 1.0 + rage * 0.18;
    col = mix(col, pale * (1.0 + rage * 0.5),
              ell(q, vec2(0.0, 0.078), vec2(0.046 * beatPuff, 0.034 * beatPuff), 0.008));

    // Both arms drum the chest, in antiphase. The drum phase is integrated on
    // the CPU -- audio modulates its *rate*, never its value -- so the motion
    // stays continuous through a beat re-trigger instead of jump-cutting, and
    // the fist travels an up-and-down arc off a flexing elbow rather than the
    // end of one stick sliding sideways.
    float drum = 0.5 - 0.5 * cos(iKongPound);
    float aL0 = mix(-0.50, -1.00, drum);
    float aL1 = mix( 0.75,  0.05, drum);
    col = mix(col, fur * 0.95, kongArm(q, vec2(-0.062, 0.105), aL0, aL1, 0.013));

    // The right arm drums too until a barrel goes out, then it sweeps up and
    // over on the throw envelope and follows through back down to the drum.
    float drumR = 1.0 - drum;
    float thr = clamp(throwT, 0.0, 1.0);
    float aR0 = mix(PI - mix(-0.50, -1.00, drumR), 1.35, thr);
    float aR1 = mix(PI - mix( 0.75,  0.05, drumR), 2.40, thr);
    col = mix(col, fur * 0.95, kongArm(q, vec2(0.062, 0.105), aR0, aR1, 0.013));

    col = mix(col, fur * 1.12, ell(q, vec2(0.0, 0.140), vec2(0.050, 0.040), 0.008));
    col = mix(col, pale, ell(q, vec2(0.0, 0.124), vec2(0.033, 0.021), 0.006));
    // Nostrils, eyes, and an angry brow.
    col = mix(col, vec3(0.20, 0.10, 0.05),
              ell(q, vec2(-0.010, 0.126), vec2(0.005, 0.004), 0.002)
            + ell(q, vec2( 0.010, 0.126), vec2(0.005, 0.004), 0.002));
    float eyes = ell(q, vec2(-0.019, 0.152), vec2(0.011, 0.010), 0.004)
               + ell(q, vec2( 0.019, 0.152), vec2(0.011, 0.010), 0.004);
    col = mix(col, vec3(1.0, 0.97, 0.92), clamp(eyes, 0.0, 1.0));
    float pupil = ell(q, vec2(-0.017, 0.150), vec2(0.005, 0.006), 0.002)
                + ell(q, vec2( 0.021, 0.150), vec2(0.005, 0.006), 0.002);
    col = mix(col, vec3(0.05, 0.02, 0.0), clamp(pupil, 0.0, 1.0));
    float brow = smoothstep(0.006, 0.0, segDist(q, vec2(-0.032, 0.168), vec2(-0.006, 0.161)))
               + smoothstep(0.006, 0.0, segDist(q, vec2( 0.032, 0.168), vec2( 0.006, 0.161)));
    col = mix(col, fur * 0.55, clamp(brow, 0.0, 1.0));
    col += vec3(1.0, 0.35, 0.15) * rage * 0.25 * ell(q, vec2(0.0, 0.09), vec2(0.10, 0.11), 0.05);
    return col;
}

// Damsel: FEET at q=(0,0), ~0.095 tall.
vec3 drawDamsel(vec2 q, float t) {
    if (abs(q.x) > 0.06 || q.y < -0.01 || q.y > 0.12) return vec3(0.0);
    vec3 dress = vec3(1.00, 0.30, 0.62);
    vec3 col = vec3(0.0);
    float fret = sin(t * 3.0) * 0.006;   // anxious sway
    q.x -= fret;
    col = mix(col, vec3(1.0, 0.80, 0.62), ell(q, vec2(-0.007, 0.008), vec2(0.006, 0.009), 0.003)
                                        + ell(q, vec2( 0.007, 0.008), vec2(0.006, 0.009), 0.003));
    // Triangular skirt built from a widening ellipse stack.
    col = mix(col, dress, ell(q, vec2(0.0, 0.030), vec2(0.024, 0.020), 0.006));
    col = mix(col, dress, ell(q, vec2(0.0, 0.052), vec2(0.016, 0.016), 0.005));
    float wave = sin(t * 5.0) * 0.5 + 0.5;   // waving for help
    float arm = smoothstep(0.006, 0.0,
                           segDist(q, vec2(0.012, 0.060), vec2(0.026, 0.072 + wave * 0.014)));
    col = mix(col, vec3(1.0, 0.80, 0.62), arm);
    col = mix(col, vec3(1.0, 0.80, 0.62), ell(q, vec2(0.0, 0.079), vec2(0.014, 0.014), 0.004));
    col = mix(col, vec3(1.0, 0.85, 0.25), ell(q, vec2(0.0, 0.090), vec2(0.017, 0.011), 0.004));
    col = mix(col, vec3(1.0, 0.85, 0.25), ell(q, vec2(-0.014, 0.072), vec2(0.007, 0.016), 0.004));
    return col;
}

vec3 drawBarrel(vec2 q, float spin, float fire, vec3 col, float t) {
    if (abs(q.x) > 0.055 || abs(q.y) > 0.055) return vec3(0.0);
    float rad = 0.036;
    float d = length(q);
    if (d > rad * 1.5) return vec3(0.0);
    float body = smoothstep(0.004, -0.004, d - rad);
    vec3 base = mix(col, vec3(1.0, 0.45, 0.10), fire);

    // Rolling read: staves spin with the barrel, so the roll direction is
    // legible instead of the barrel looking like a static glowing blob.
    vec2 sq = rot(spin) * q;
    float ang = atan(sq.y, sq.x);
    float staves = smoothstep(0.35, 0.0, abs(fract(ang / TAU * 6.0 + 0.5) - 0.5) * 6.0 - 0.35);
    float hoop = smoothstep(0.0045, 0.0, abs(d - rad * 0.62));

    vec3 o = base * body * 0.45;
    o += base * 1.5 * body * staves * 0.35;
    o += base * 1.7 * hoop * body;
    o += base * 2.0 * smoothstep(0.005, 0.0, abs(d - rad)) * body;   // bright rim
    if (fire > 0.5) {
        float flick = hash21(floor(vec2(t * 22.0, floor(ang * 3.0))));
        o += vec3(1.0, 0.55, 0.12) * body * (0.30 + flick * 0.65);
        o += vec3(1.0, 0.30, 0.05) * smoothstep(rad * 1.5, rad, d) * 0.35;
    }
    return o;
}

vec3 drawHammer(vec2 q, float t) {
    if (abs(q.x) > 0.045 || abs(q.y) > 0.045) return vec3(0.0);
    float bob = sin(t * 4.0) * 0.004;
    q.y -= bob;
    float shaft = smoothstep(0.005, 0.0, segDist(q, vec2(0.0, -0.020), vec2(0.0, 0.012)));
    float head = ell(q, vec2(0.0, 0.020), vec2(0.020, 0.013), 0.005);
    vec3 o = vec3(0.55, 0.30, 0.12) * shaft;
    o += vec3(1.0, 0.85, 0.25) * head;
    o += vec3(1.0, 0.7, 0.2) * head * (0.4 + 0.4 * sin(t * 6.0));
    return o;
}

// Burning oil drum at the foot of the rig.
vec3 drawOil(vec2 q, float t) {
    if (abs(q.x) > 0.06 || q.y < -0.01 || q.y > 0.12) return vec3(0.0);
    vec3 o = vec3(0.0);
    float drum = ell(q, vec2(0.0, 0.026), vec2(0.030, 0.028), 0.006);
    o = mix(o, vec3(0.25, 0.35, 0.55), drum);
    o += vec3(0.3, 0.5, 0.8) * smoothstep(0.005, 0.0, abs(abs(q.y - 0.026) - 0.014)) * drum;
    float fl = hash21(floor(vec2(t * 18.0, 3.0)));
    float flame = ell(q, vec2(0.0, 0.064 + fl * 0.008),
                      vec2(0.019, 0.026 + fl * 0.010), 0.014);
    o += mix(vec3(1.0, 0.35, 0.05), vec3(1.0, 0.85, 0.25), fl) * flame * (1.1 + iBass * 0.6);
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

    col += drawOil(p - iOilPos, t);

    if (iHammerAlive > 0.5) col += drawHammer(p - iHammerPos, t);

    for (int i = 0; i < 8; i++) {
        if (iBarrelAlive[i] < 0.5) continue;
        col += drawBarrel(p - iBarrelPos[i], iBarrelRot[i], iBarrelFire[i],
                          mix(vec3(0.85, 0.5, 0.2), palette(iHue + 0.5), 0.3), t)
             * (1.0 + iBass * 0.3);
    }

    // Smash bursts: expanding spark ring plus flung debris where a hammer
    // blow landed. Fades on a curve rather than linearly so the first
    // instant reads as a hard hit instead of a slow smoke ring.
    for (int i = 0; i < 10; i++) {
        if (iSmash[i].z < 0.0) continue;
        float age = iSmash[i].z;
        vec2 d = p - iSmash[i].xy;
        float dl = length(d);
        float fade = pow(1.0 - age, 1.6);
        float rad = 0.010 + age * 0.060;
        float ring = smoothstep(0.010, 0.0, abs(dl - rad)) * fade;
        float ang = atan(d.y, d.x);
        float spokes = pow(0.5 + 0.5 * sin(ang * 8.0 + float(i) * 2.3), 6.0);
        vec3 hot = mix(vec3(1.0, 0.85, 0.3), palette(iHue + 0.15), 0.3);
        col += hot * ring * (1.6 + spokes * 2.2) * (1.4 + iTreble * 0.9);
        col += hot * exp(-dl * dl * 900.0) * fade * 2.2;              // core pop
        // Debris chunks flung outward along the spokes.
        for (int k = 0; k < 5; k++) {
            float fk = float(k);
            float a2 = fk / 5.0 * TAU + hash21(vec2(fk, float(i))) * 6.28;
            vec2 dp = d - vec2(cos(a2), sin(a2)) * (rad * 1.15);
            col += hot * exp(-dot(dp, dp) * 5200.0) * fade * 1.5;
        }
    }

    float heroVis = (iHeroHit > 0.5) ? step(0.5, fract(t * 14.0)) : 1.0;
    col += drawHero(p - iHeroPos, sign(iHeroFace + 0.001), iHeroClimb, iHeroHammer, t)
         * heroVis * (1.0 + iMid * 0.2);

    col += drawKong(p - iKongPos, iKongThrow, iKongRage) * (1.0 + iBeat * 0.25);
    col += drawDamsel(p - iDamselPos, t);

    // Victory hearts rise between the rescued pair.
    if (iWin > 0.0) {
        vec2 mid = mix(iKongPos, iDamselPos, 0.5) + vec2(0.0, 0.045);
        for (int i = 0; i < 4; i++) {
            float fi = float(i);
            float ph = fract((1.0 - iWin) * 0.9 + fi * 0.25);
            vec2 hp = mid + vec2(sin(ph * 6.0 + fi) * 0.05, ph * 0.10);
            vec2 hq = (p - hp) / 0.028;
            hq.y -= 0.35;
            float hs = pow(hq.x * hq.x + hq.y * hq.y - 1.0, 3.0) - hq.x * hq.x * hq.y * hq.y * hq.y;
            float m = smoothstep(0.0, -0.7, hs) * iWin * (1.0 - ph);
            col += vec3(1.0, 0.30, 0.50) * m * 2.2;
        }
    }

    col += palette(iHue + 0.5) * iFlash * 0.22;
    // Sparkle, same treatment: round motes, not filled cells.
    vec2 spc = p * iResolution.y * 0.25;
    vec2 spg = floor(spc) + floor(t * 20.0);
    float sp = hash21(spg);
    if (sp > 0.985) {
        vec2 spd = fract(spc) - vec2(hash21(spg + 2.7), hash21(spg + 6.1));
        col += vec3(0.9, 0.85, 1.0) * exp(-dot(spd, spd) * 30.0)
             * (0.35 + iTreble * 0.8) * iGlitter;
    }

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

    # Progress-fraction per second along a girder. At 0.55 the hero crossed
    # a whole girder in under two seconds, which read as a sprint and left
    # no time to see him react to anything.
    _HERO_SPEED = 0.30
    _CLIMB_SPEED = 0.9
    _BARREL_SPEED = 0.32
    _HAMMER_TIME = 4.5         # seconds the hammer stays up once grabbed

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
        # The simulation keeps x as a [0,1] screen fraction and only converts
        # to the shader's centred world x at upload (see _wx) — a fixed world
        # span would leave the rig floating in the middle of any window wider
        # than square, which is exactly the bug that stranded the Missile
        # Command skyline.  y is already aspect-independent, so it stays world.
        # Even rows descend to the right, odd rows descend to the left, so a
        # barrel released at Kong's feet zig-zags all the way down instead of
        # rolling straight off one end.
        # The board is laid out fresh on every activation. It used to be a
        # single hard-coded arrangement -- same pitch, same spacing, ladders
        # always at 0.30/0.70 -- so every run opened on an identical screen,
        # which the pack's randomisation rule exists to prevent. What varies:
        # which way the zig-zag starts, how steeply the girders pitch, how far
        # apart the rows sit, and where the ladders land.
        left, right = 0.0, 1.0
        span = right - left
        flip = self.rng.random() < 0.5
        pitch = float(self.rng.uniform(0.038, 0.075))
        gap = float(self.rng.uniform(0.140, 0.172))
        base = -0.44 + float(self.rng.uniform(-0.015, 0.030))
        self._rows: list[tuple[float, float, float, float]] = []
        for i in range(_NROWS):
            y = base + i * gap
            # A girder descends one way or the other; the parity alternates so
            # a barrel released at Kong's feet zig-zags all the way down.
            if (i % 2 == 0) != flip:
                self._rows.append((left, y + pitch, right, y))
            else:
                self._rows.append((left, y, right, y + pitch))

        # One or two ladders per gap, jittered. Every gap keeps at least one,
        # or the hero would have no way up and would stall on that girder.
        self._ladders: list[dict] = []
        for i in range(_NROWS - 1):
            if self.rng.random() < 0.30:
                fracs = [float(self.rng.uniform(0.30, 0.70))]
            else:
                fracs = [float(self.rng.uniform(0.16, 0.40)),
                         float(self.rng.uniform(0.60, 0.84))]
                if self.rng.random() < 0.5:
                    fracs.reverse()
            for fr in fracs:
                self._ladders.append({'row_lo': i, 'row_hi': i + 1,
                                      'x': left + span * fr})

        self._hero_row = 0
        self._hero_t = float(self.rng.uniform(0.10, 0.22))
        self._hero_climb: dict | None = None
        self._hero_climb_frac = 0.0
        self._hero_face = 1.0
        self._hero_jump = 0.0
        self._hero_hit = 0.0
        self._hero_hammer = 0.0

        self._kong_pos = self._row_point(_NROWS - 1, float(self.rng.uniform(0.07, 0.22)))
        self._kong_throw = 0.0
        self._kong_rage = 0.0
        self._kong_pound = float(self.rng.uniform(0.0, 2.0 * math.pi))
        self._throw_cd = float(self.rng.uniform(1.0, 2.5))
        self._damsel_pos = self._row_point(_NROWS - 1, float(self.rng.uniform(0.78, 0.93)))
        self._oil_pos = self._row_point(0, float(self.rng.uniform(0.90, 0.99)))

        self._barrels: list[dict] = []
        self._hammer: dict | None = None
        self._hammer_cd = float(self.rng.uniform(6.0, 14.0))
        self._smash = np.full((_NSMASH, 3), -1.0, dtype=np.float32)
        self._win = 0.0

    # ------------------------------------------------------------- geometry
    def _row_point(self, row: int, tfrac: float) -> tuple[float, float]:
        x0, y0, x1, y1 = self._rows[row]
        tfrac = float(np.clip(tfrac, 0.0, 1.0))
        return (x0 + (x1 - x0) * tfrac, y0 + (y1 - y0) * tfrac)

    def _row_downhill_dir(self, row: int) -> float:
        """+1 if the row descends toward t=1 (low end on the right), else -1."""
        _, y0, _, y1 = self._rows[row]
        return 1.0 if y1 < y0 else -1.0

    def _t_at_x(self, row: int, x: float) -> float:
        x0, _, x1, _ = self._rows[row]
        return float(np.clip((x - x0) / max(x1 - x0, 1e-6), 0.0, 1.0))

    def _ladder_y(self, lad: dict) -> tuple[float, float]:
        """(bottom_y, top_y) of a ladder in world space."""
        lo, hi = lad['row_lo'], lad['row_hi']
        y0 = self._row_point(lo, self._t_at_x(lo, lad['x']))[1]
        y1 = self._row_point(hi, self._t_at_x(hi, lad['x']))[1]
        return (y0, y1)

    # ------------------------------------------------------------------ AI
    def _nearest_up_ladder(self, row: int, t: float) -> dict | None:
        cands = [ladder for ladder in self._ladders if ladder['row_lo'] == row]
        if not cands:
            return None
        x0, _, x1, _ = self._rows[row]
        cur_x = x0 + (x1 - x0) * t
        return min(cands, key=lambda ladder: abs(ladder['x'] - cur_x))

    def _hero_pos(self) -> tuple[float, float]:
        if self._hero_climb is not None:
            y0, y1 = self._ladder_y(self._hero_climb)
            return (self._hero_climb['x'], y0 + (y1 - y0) * self._hero_climb_frac)
        x, y = self._row_point(self._hero_row, self._hero_t)
        if self._hero_jump > 0.0:
            y += 0.085 * float(np.sin(np.pi * self._hero_jump))
        return (x, y)

    def _barrel_near(self, row: int, t: float) -> bool:
        """True when a barrel is rolling toward the hero and close enough to jump.

        The old check looked only in the direction the hero was *walking*, so a
        barrel closing on him from behind never triggered a jump -- and since
        barrels run downhill while the hero is usually working toward the high
        end of a girder, that was most of them. What matters is whether the
        barrel is closing, which is what the girder's downhill direction says.
        """
        down = self._row_downhill_dir(row)
        for b in self._barrels:
            if b['row'] != row or b['lad'] is not None:
                continue
            if (t - b['t']) * down > 0.0 and abs(b['t'] - t) < 0.13:
                return True
        return False

    def _spawn_smash(self, x: float, y: float) -> None:
        free = np.where(self._smash[:, 2] < 0.0)[0]
        slot = int(free[0]) if free.size else int(np.argmax(self._smash[:, 2]))
        self._smash[slot] = (x, y, 0.0)

    def _update_hero(self, dt: float, tempo: float) -> None:
        self._hero_jump = max(0.0, self._hero_jump - dt * 1.7)

        if self._hero_climb is not None:
            self._hero_climb_frac += dt * self._CLIMB_SPEED * tempo
            if self._hero_climb_frac >= 1.0:
                self._hero_row = self._hero_climb['row_hi']
                self._hero_t = self._t_at_x(self._hero_row, self._hero_climb['x'])
                self._hero_climb = None
                if self._hero_row == _NROWS - 1:
                    self._win = 1.0
                    self._flash = max(self._flash, 0.8)
                    self._hue = float((self._hue + self.rng.uniform(0.1, 0.25)) % 1.0)
                    self._hero_row = 0
                    self._hero_t = float(self.rng.uniform(0.10, 0.22))
            return

        # With the hammer up the hero stops climbing and goes hunting — the
        # classic trade-off, and it reads clearly on screen.
        if self._hero_hammer > 0.0:
            if self._barrel_near(self._hero_row, self._hero_t) and self._hero_jump <= 0.0:
                self._hero_jump = 1.0
            prey = [b for b in self._barrels
                    if b['row'] == self._hero_row and b['lad'] is None]
            if prey:
                nearest = min(prey, key=lambda b: abs(b['t'] - self._hero_t))
                ahead = 1.0 if nearest['t'] > self._hero_t else -1.0
                self._hero_face = ahead
                self._hero_t = float(np.clip(
                    self._hero_t + ahead * self._HERO_SPEED * tempo * dt, 0.0, 1.0))
            for b in list(self._barrels):
                if (b['row'] == self._hero_row and b['lad'] is None
                        and abs(b['t'] - self._hero_t) < 0.05):
                    bx, by = self._row_point(b['row'], b['t'])
                    self._spawn_smash(bx, by)
                    self._barrels.remove(b)
                    self._flash = max(self._flash, 0.35)
            return

        target: dict | None = None
        if self._hammer is not None and self._hammer['row'] == self._hero_row:
            target = {'x': self._hammer_x(), 'grab': True}
        else:
            lad = self._nearest_up_ladder(self._hero_row, self._hero_t)
            if lad is not None:
                target = {'x': lad['x'], 'grab': False, 'ladder': lad}
        if target is None:
            return

        target_t = self._t_at_x(self._hero_row, target['x'])
        ahead = 1.0 if target_t > self._hero_t else -1.0
        self._hero_face = ahead
        if self._barrel_near(self._hero_row, self._hero_t) and self._hero_jump <= 0.0:
            self._hero_jump = 1.0
        step = ahead * self._HERO_SPEED * tempo * dt
        if abs(target_t - self._hero_t) < abs(step):
            self._hero_t = target_t
            if target['grab']:
                self._hammer = None
                self._hero_hammer = self._HAMMER_TIME
                self._flash = max(self._flash, 0.5)
            else:
                self._hero_climb = target['ladder']
                self._hero_climb_frac = 0.0
        else:
            self._hero_t += step

        if self._hero_hit <= 0.0 and self._hero_jump <= 0.0:
            for b in self._barrels:
                if (b['row'] == self._hero_row and b['lad'] is None
                        and abs(b['t'] - self._hero_t) < 0.025):
                    self._hero_hit = 1.0
                    self._flash = max(self._flash, 0.5)
                    self._hero_row = 0
                    self._hero_t = float(self.rng.uniform(0.10, 0.22))
                    self._hero_climb = None
                    break

    def _hammer_x(self) -> float:
        assert self._hammer is not None
        return self._row_point(self._hammer['row'], self._hammer['t'])[0]

    def _update_barrels(self, dt: float, tempo: float) -> None:
        self._throw_cd -= dt
        if self._throw_cd <= 0.0 and len(self._barrels) < _NBARREL:
            row = _NROWS - 1
            t0 = 0.14 if self._row_downhill_dir(row) > 0 else 0.86
            self._barrels.append({
                'row': row, 't': t0, 'rot': 0.0, 'lad': None, 'ladf': 0.0,
                'fire': 1.0 if self.rng.random() < 0.25 else 0.0,
            })
            self._kong_throw = 1.0
            self._throw_cd = float(self.rng.uniform(1.6, 3.0)) / (1.0 + self._bass * 0.5)

        for b in list(self._barrels):
            if b['lad'] is not None:
                b['ladf'] += dt * 0.9 * tempo
                b['rot'] += dt * tempo * 3.0
                if b['ladf'] >= 1.0:
                    lad = b['lad']
                    b['row'] = lad['row_lo']
                    b['t'] = self._t_at_x(lad['row_lo'], lad['x'])
                    b['lad'] = None
                    b['ladf'] = 0.0
                continue

            d = self._row_downhill_dir(b['row'])
            prev_x = self._row_point(b['row'], b['t'])[0]
            b['t'] += d * dt * self._BARREL_SPEED * tempo
            b['rot'] += dt * self._BARREL_SPEED * tempo * 6.0
            new_x = self._row_point(b['row'], float(np.clip(b['t'], 0.0, 1.0)))[0]

            # Crossing a down-ladder: sometimes take it instead of rolling on.
            if b['row'] > 0:
                for lad in self._ladders:
                    if lad['row_hi'] != b['row']:
                        continue
                    lx = lad['x']
                    if (prev_x - lx) * (new_x - lx) <= 0.0 and self.rng.random() < 0.28:
                        b['lad'] = lad
                        b['ladf'] = 0.0
                        b['t'] = self._t_at_x(b['row'], lx)
                        break
            if b['lad'] is not None:
                continue

            if b['t'] < 0.0 or b['t'] > 1.0:
                if b['row'] == 0:
                    self._barrels.remove(b)
                else:
                    b['row'] -= 1
                    b['t'] = float(np.clip(b['t'], 0.0, 1.0))

    def _update_hammer(self, dt: float) -> None:
        self._hero_hammer = max(0.0, self._hero_hammer - dt)
        if self._hammer is not None or self._hero_hammer > 0.0:
            return
        self._hammer_cd -= dt
        if self._hammer_cd <= 0.0:
            row = int(self.rng.integers(0, _NROWS - 1))
            self._hammer = {'row': row, 't': float(self.rng.uniform(0.25, 0.75))}
            self._hammer_cd = float(self.rng.uniform(10.0, 18.0))

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
            self._kong_rage = 1.0
        self._beat = max(0.0, self._beat - dt * 2.5)
        self._flash = max(0.0, self._flash - dt * 1.8)
        self._hero_hit = max(0.0, self._hero_hit - dt * 1.4)
        self._kong_throw = max(0.0, self._kong_throw - dt * 1.6)
        self._kong_rage = max(0.0, self._kong_rage - dt * 1.9)
        # Kong idles at about one beat of the chest a second and drums roughly
        # three times faster while the beat envelope is up. Advancing the phase
        # keeps it continuous however the envelope moves.
        self._kong_pound = (self._kong_pound
                            + dt * (6.3 + self._kong_rage * 13.0)) % (2.0 * math.pi)
        self._win = max(0.0, self._win - dt * 0.55)

        tempo = max(0.05, float(self.parameters['speed'])) * (1.0 + self._mid * 0.4)
        self._update_hammer(dt)
        self._update_hero(dt, tempo)
        self._update_barrels(dt, tempo * (1.0 + self._bass * 0.3))

        act = self._smash[:, 2] >= 0.0
        self._smash[act, 2] += dt * 1.9
        self._smash[self._smash[:, 2] > 1.0] = -1.0

    # ---------------------------------------------------------------- render
    def _barrel_world(self, b: dict) -> tuple[float, float]:
        if b['lad'] is not None:
            y0, y1 = self._ladder_y(b['lad'])
            return (b['lad']['x'], y1 + (y0 - y1) * b['ladf'])
        return self._row_point(b['row'], b['t'])

    @staticmethod
    def _wx(fx: float, aspect: float) -> float:
        """Convert a [0,1] screen-fraction x to the shader's centred world x."""
        return (fx - 0.5) * aspect

    def _wpt(self, pt: tuple[float, float], aspect: float) -> tuple[float, float]:
        return (self._wx(pt[0], aspect), pt[1])

    def render(self) -> None:
        """Upload the climb state and draw the neon rig."""
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
        p['iWin'].value = float(self._win)
        p['iRow'].value = [
            (self._wx(x0, aspect), float(y0), self._wx(x1, aspect), float(y1))
            for x0, y0, x1, y1 in self._rows
        ]

        ladder_val = []
        for ladder in self._ladders:
            y0, y1 = self._ladder_y(ladder)
            ladder_val.append((self._wx(ladder['x'], aspect), float(y0), float(y1), 0.0))
        while len(ladder_val) < _NLADDER:
            ladder_val.append((_OFF, _OFF, _OFF, 0.0))
        p['iLadder'].value = ladder_val
        p['iLadderCount'].value = len(self._ladders)

        p['iHeroPos'].value = self._wpt(self._hero_pos(), aspect)
        p['iHeroFace'].value = float(self._hero_face)
        p['iHeroClimb'].value = 1.0 if self._hero_climb is not None else 0.0
        p['iHeroHit'].value = 1.0 if self._hero_hit > 0.0 else 0.0
        p['iHeroHammer'].value = float(self._hero_hammer)

        p['iKongPos'].value = self._wpt(self._kong_pos, aspect)
        p['iKongThrow'].value = float(self._kong_throw)
        p['iKongRage'].value = float(self._kong_rage)
        p['iKongPound'].value = float(self._kong_pound)
        p['iDamselPos'].value = self._wpt(self._damsel_pos, aspect)
        p['iOilPos'].value = self._wpt(self._oil_pos, aspect)

        bpos, brot, balive, bfire = [], [], [], []
        for i in range(_NBARREL):
            if i < len(self._barrels):
                b = self._barrels[i]
                bpos.append(self._wpt(self._barrel_world(b), aspect))
                brot.append(float(b['rot']))
                balive.append(1.0)
                bfire.append(float(b['fire']))
            else:
                bpos.append((_OFF, _OFF))
                brot.append(0.0)
                balive.append(0.0)
                bfire.append(0.0)
        p['iBarrelPos'].value = bpos
        p['iBarrelRot'].value = brot
        p['iBarrelAlive'].value = balive
        p['iBarrelFire'].value = bfire

        if self._hammer is not None:
            hx, hy = self._row_point(self._hammer['row'], self._hammer['t'])
            p['iHammerPos'].value = (self._wx(hx, aspect), hy + 0.04)
            p['iHammerAlive'].value = 1.0
        else:
            p['iHammerPos'].value = (_OFF, _OFF)
            p['iHammerAlive'].value = 0.0

        p['iSmash'].value = [
            (self._wx(float(s[0]), aspect), float(s[1]), float(s[2])) if s[2] >= 0.0
            else (_OFF, _OFF, -1.0)
            for s in self._smash
        ]

        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()

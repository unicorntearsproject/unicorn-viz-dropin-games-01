"""Q*bert — a real isometric pyramid-hopping simulation rendered as a neon
light-show.

Same architecture as the rest of the pack: an actual game runs on the CPU and
the shader renders the true state. The full cast is here, each a real actor in
the simulation rather than set dressing:

* **Q*bert** — orange, tubular snout that swings to face his hop direction,
  big eyes that track, little feet. Hops diagonally toward the nearest cube he
  still needs, steering around whatever is currently dangerous.
* **Coily** — the purple snake. Spawns as an egg at the apex, bounces down,
  hatches into a coiled spring-bodied snake with a flicking forked tongue, and
  then hunts Q*bert across the pyramid.
* **Red balls** — bounce down the pyramid on their own, lethal on contact.
* **Slick & Sam** — little green creatures that hop downward *un-colouring*
  cubes Q*bert already finished, so a course can be actively undone.
* **Flying discs** — one either side. Corner Q*bert with Coily adjacent and he
  leaps onto a disc; Coily dives off the edge after him and the disc is spent.

**Courses.** Six of them, cycled without immediate repeats: pyramids of 5–8
rows in three rulesets — ``once`` (one hop per cube), ``two`` (each cube needs
two visits through an intermediate colour), and ``toggle`` (landing flips a
cube back off, so Q*bert has to avoid his own finished work). Each course also
carries its own cube palette, so a switch is unmistakable.

**Beat pulse.** Every beat sends a wave down the pyramid: cubes lift and flare
in sequence from the apex outward. The lift is computed once on the CPU and
baked into the uploaded cube position, then applied to whatever is standing on
that row too, so characters ride the wave with their platform instead of
detaching from it.

Audio reactivity:
  - bass    -> hop speed, cube swell, edge-glow gain
  - mid     -> enemy hop rate, palette drift, disc shimmer
  - treble  -> cube-face sparkle, drip-streak sparkle
  - beat    -> platform pulse wave, screen flash, palette shift

Randomised per activation: course, palette hue, hero starting phase, enemy and
hazard timers, and ``self.time`` — no two runs look or play alike from frame 0.
"""
from __future__ import annotations

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

_NCUBE_MAX = 36          # 8 rows: 1+2+...+8
_NBALL = 4
_NSLICK = 2
_NDISC = 2
_OFF = 1000.0

# Cube lattice per row count: (dx, dy). Chosen so the tallest course still
# leaves headroom for the character standing on the apex and clears the
# bottom edge, at both 16:9 and 4:3.
_GEOM: dict[int, tuple[float, float]] = {
    5: (0.193, 0.150), 6: (0.166, 0.129), 7: (0.146, 0.114), 8: (0.130, 0.101),
}
# In half-width units: how far a cube's side skirt hangs below its top face,
# and how tall a character stands above the face it is on. Both feed the
# vertical centring in _build_course.
_CUBE_DEPTH = 1.138      # hh + side, matching the shader's drawCube
_CHAR_H = 2.33

# Course roster: pyramid size + colouring rule + cube palette.
#   'once'   — one landing finishes a cube
#   'two'    — two landings, through an intermediate colour
#   'toggle' — landing flips the cube back off again
_COURSES: tuple[dict, ...] = (
    {'rows': 7, 'mode': 'once', 'a': (0.16, 0.10, 0.38), 'b': (1.00, 0.75, 0.15)},
    {'rows': 5, 'mode': 'two', 'a': (0.05, 0.22, 0.34), 'b': (0.20, 1.00, 0.85)},
    {'rows': 6, 'mode': 'once', 'a': (0.30, 0.06, 0.24), 'b': (1.00, 0.35, 0.70)},
    {'rows': 8, 'mode': 'once', 'a': (0.10, 0.14, 0.40), 'b': (0.55, 0.95, 1.00)},
    {'rows': 6, 'mode': 'toggle', 'a': (0.34, 0.14, 0.03), 'b': (1.00, 0.85, 0.30)},
    {'rows': 7, 'mode': 'two', 'a': (0.06, 0.24, 0.12), 'b': (0.60, 1.00, 0.30)},
)

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
uniform float iPulse;           // beat wave envelope [0,1]
uniform float iRows;
uniform float iCubeHW;          // cube half-width; scales with course size
uniform float iSprite;          // character scale, relative to the 7-row course
uniform vec4  iCube[36];        // (x, y, state01, row) — y already pulse-lifted
uniform int   iCubeCount;
uniform vec3  iColA;            // course start colour
uniform vec3  iColB;            // course target colour
uniform vec2  iHeroPos;
uniform float iHeroFace;
uniform float iHeroHop;
uniform float iHeroHit;
uniform float iHeroRide;        // >0 while riding a disc
uniform vec2  iCoilyPos;
uniform float iCoilyHop;
uniform float iCoilyAlive;
uniform float iCoilyEgg;        // 1 while still an unhatched egg
uniform vec3  iBall[4];         // (x, y, alive)
uniform vec3  iSlick[2];        // (x, y, alive)
uniform vec3  iDisc[2];         // (x, y, alive)
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
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }
float ell(vec2 q, vec2 c, vec2 r, float soft) {
    return smoothstep(soft, -soft, length((q - c) / r) - 1.0);
}
float segDist(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}
// Paint one sprite layer, tracking coverage alongside colour so characters can
// be composited over the pyramid at full opacity instead of added to it (an
// additive sprite lets the cube underneath show through and wash it out).
void paint(inout vec3 o, inout float cov, vec3 c, float m) {
    m = clamp(m, 0.0, 1.0);
    o = mix(o, c, m);
    cov = max(cov, m);
}

vec3 backdrop(vec2 uv, float A, float tb) {
    vec2 p = (uv - 0.5) * vec2(A, 1.0);
    float v = sin(p.x * 2.2 + tb * 0.3) + sin(p.y * 1.8 - tb * 0.24);
    vec3 col = palette(v * 0.08 + iHue + 0.6 + tb * 0.014);
    col *= col;
    col *= 0.05 + 0.05 * (v * 0.5 + 0.5);
    float sh = hash21(floor(uv * vec2(A, 1.0) * 100.0));
    if (sh > 0.972) {
        float tw = 0.5 + 0.5 * sin(tb * (2.0 + sh * 6.0) + sh * 40.0);
        col += vec3(0.6, 0.7, 0.9) * tw * (0.3 + iTreble * 0.4);
    }
    return col;
}

// Isometric cube: diamond top face plus two side faces, with a glossy
// highlight on top and neon paint dripping down the sides.
// Works in half-width-normalised space so one routine serves every course
// size — the caller only supplies the cube half-width.
vec3 drawCube(vec2 qw, vec3 topCol, float glow, float tb, float seed, float hw) {
    vec2 q = qw / hw;
    const float hh = 0.586, side = 0.552;
    if (abs(q.x) > 1.08 || q.y > hh + 0.10 || q.y < -hh - side - 0.10)
        return vec3(0.0);
    vec3 o = vec3(0.0);

    float diamond = abs(q.x) + abs(q.y) / hh;
    float top = smoothstep(1.03, 0.97, diamond);

    // Side faces hang off the lower-left and lower-right top edges.
    float lyEdge = -hh * (q.x + 1.0);
    float lf = step(-1.0, q.x) * step(q.x, 0.0)
             * step(lyEdge - side, q.y) * step(q.y, lyEdge);
    float ryEdge = -hh * (1.0 - q.x);
    float rf = step(0.0, q.x) * step(q.x, 1.0)
             * step(ryEdge - side, q.y) * step(q.y, ryEdge);

    o = mix(o, topCol * (0.95 + glow * 0.5), top);
    o = mix(o, topCol * (0.30 + glow * 0.22), lf);
    o = mix(o, topCol * (0.52 + glow * 0.30), rf);

    // Gloss sweep across the top face.
    o += vec3(1.0) * top * 0.30
       * smoothstep(1.0, 0.0, length(q - vec2(-0.30, hh * 0.32)) / 0.52);

    // Neon paint dripping down the side faces.
    float lanes = 12.0;
    float lane = floor((q.x + 1.0) * lanes);
    float h = hash21(vec2(lane, seed));
    float flow = fract(tb * (0.12 + h * 0.25) + h);
    float dripTop = (q.x < 0.0 ? lyEdge : ryEdge);
    float dripLen = 0.14 + h * 0.34;
    float dy = dripTop - q.y;
    float drip = smoothstep(0.30, 0.0, abs(fract((q.x + 1.0) * lanes) - 0.5))
               * step(0.0, dy) * smoothstep(dripLen * flow + 0.06, 0.0, dy);
    o += topCol * drip * clamp(lf + rf, 0.0, 1.0) * 1.4 * (0.6 + iTreble * 0.7);

    // Neon rim on the top-face outline.
    o += topCol * smoothstep(0.12, 0.0, abs(diamond - 1.0)) * top
       * (0.7 + glow * 0.8 + iBass * 0.4);
    return o;
}

// Q*bert: orange, snout swings to the facing side, eyes track with it.
vec4 drawQbert(vec2 q, float face, float hop, float tb) {
    if (abs(q.x) > 0.062 || q.y < -0.012 || q.y > 0.095) return vec4(0.0);
    vec3 orange = vec3(1.00, 0.52, 0.08);
    vec3 o = vec3(0.0);
    float cov = 0.0;
    float squash = 1.0 - sin(hop * PI) * 0.12;   // stretches through the arc

    float feet = ell(q, vec2(-0.015, 0.007), vec2(0.013, 0.008), 0.004)
               + ell(q, vec2(0.015, 0.007), vec2(0.013, 0.008), 0.004);
    paint(o, cov, vec3(1.0, 0.78, 0.20), feet);

    float body = ell(q, vec2(0.0, 0.042), vec2(0.031, 0.033 * squash), 0.006);
    paint(o, cov, orange, body);

    // Signature tubular snout.
    vec2 sq = q - vec2(face * 0.028, 0.036);
    paint(o, cov, orange * 0.92, ell(sq, vec2(face * 0.006, 0.0), vec2(0.019, 0.010), 0.005));
    paint(o, cov, vec3(0.12, 0.05, 0.02),
          ell(sq, vec2(face * 0.019, 0.0), vec2(0.006, 0.007), 0.003));

    float eyes = ell(q, vec2(-0.011, 0.064), vec2(0.013, 0.014), 0.004)
               + ell(q, vec2(0.011, 0.064), vec2(0.013, 0.014), 0.004);
    paint(o, cov, vec3(1.0, 0.98, 0.95), eyes);
    float pup = ell(q, vec2(-0.009 + face * 0.005, 0.062), vec2(0.006, 0.007), 0.002)
              + ell(q, vec2(0.013 + face * 0.005, 0.062), vec2(0.006, 0.007), 0.002);
    paint(o, cov, vec3(0.05, 0.02, 0.02), pup);

    // Highlights ride on top of already-covered pixels, so they brighten the
    // sprite without needing to extend its coverage.
    o += vec3(1.0) * ell(q, vec2(-0.013, 0.052), vec2(0.009, 0.007), 0.007) * 0.35;
    o += orange * body * 0.25 * (0.5 + 0.5 * sin(tb * 5.0));
    return vec4(o, cov);
}

// Coily: purple coiled snake, or the egg he arrives in.
vec4 drawCoily(vec2 q, float egg, float hop, float tb) {
    if (abs(q.x) > 0.055 || q.y < -0.012 || q.y > 0.125) return vec4(0.0);
    vec3 purple = vec3(0.62, 0.18, 1.00);
    vec3 o = vec3(0.0);
    float cov = 0.0;
    // The spring stretches out mid-hop and compresses on landing.
    float stretch = 1.0 + sin(hop * PI) * 0.35;
    if (egg > 0.5) {
        paint(o, cov, vec3(0.85, 0.80, 1.00),
              ell(q, vec2(0.0, 0.030), vec2(0.022 / stretch, 0.028 * stretch), 0.005));
        o += vec3(0.5, 0.3, 1.0) * ell(q, vec2(-0.007, 0.040), vec2(0.008, 0.010), 0.006) * 0.5;
        return vec4(o, cov);
    }
    // Coiled spring body — coils spread apart as it extends.
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        float wob = sin(tb * 6.0 + fi * 1.3) * 0.005;
        paint(o, cov, purple * (0.55 + fi * 0.16),
              ell(q, vec2(wob, 0.012 + fi * 0.017 * stretch),
                  vec2(0.027 - fi * 0.004, 0.011), 0.004));
    }
    vec2 hq = q - vec2(sin(tb * 6.0 + 3.9) * 0.005, 0.080 + (stretch - 1.0) * 0.030);
    paint(o, cov, purple * 1.15, ell(hq, vec2(0.0, 0.0), vec2(0.025, 0.021), 0.005));
    float eyes = ell(hq, vec2(-0.010, 0.007), vec2(0.010, 0.011), 0.003)
               + ell(hq, vec2(0.010, 0.007), vec2(0.010, 0.011), 0.003);
    paint(o, cov, vec3(1.0), eyes);
    paint(o, cov, vec3(0.95, 0.06, 0.06),
          ell(hq, vec2(-0.009, 0.006), vec2(0.005, 0.006), 0.002)
        + ell(hq, vec2(0.011, 0.006), vec2(0.005, 0.006), 0.002));
    // Forked tongue flicks in and out — extends past the head, so it carries
    // its own coverage.
    float flick = step(0.55, fract(tb * 2.2));
    paint(o, cov, vec3(1.0, 0.12, 0.25), flick
          * smoothstep(0.0035, 0.0, segDist(hq, vec2(0.0, -0.012), vec2(0.0, -0.026))));
    return vec4(o, cov);
}

// Slick / Sam: little green cube-thief with a tuft of hair.
vec4 drawSlick(vec2 q, float tb) {
    if (abs(q.x) > 0.045 || q.y < -0.012 || q.y > 0.075) return vec4(0.0);
    vec3 green = vec3(0.30, 0.95, 0.25);
    vec3 o = vec3(0.0);
    float cov = 0.0;
    paint(o, cov, green, ell(q, vec2(0.0, 0.028), vec2(0.023, 0.026), 0.005));
    float eyes = ell(q, vec2(-0.008, 0.038), vec2(0.009, 0.010), 0.003)
               + ell(q, vec2(0.008, 0.038), vec2(0.009, 0.010), 0.003);
    paint(o, cov, vec3(1.0), eyes);
    paint(o, cov, vec3(0.05, 0.05, 0.05),
          ell(q, vec2(-0.007, 0.037), vec2(0.004, 0.005), 0.002)
        + ell(q, vec2(0.009, 0.037), vec2(0.004, 0.005), 0.002));
    // Hair tuft waggles — sits outside the body, so it carries coverage.
    float w = sin(tb * 7.0) * 0.006;
    paint(o, cov, green * 1.3, smoothstep(0.005, 0.0,
                                          segDist(q, vec2(0.0, 0.050), vec2(w, 0.066))));
    return vec4(o, cov);
}

vec4 drawBall(vec2 q, float tb) {
    if (abs(q.x) > 0.030 || abs(q.y - 0.024) > 0.030) return vec4(0.0);
    float b = ell(q, vec2(0.0, 0.024), vec2(0.020, 0.020), 0.005);
    vec3 o = vec3(1.0, 0.15, 0.12);
    o += vec3(1.0, 0.75, 0.7) * ell(q, vec2(-0.007, 0.031), vec2(0.007, 0.006), 0.005) * 0.8;
    o += vec3(1.0, 0.2, 0.15) * (0.35 + 0.25 * sin(tb * 8.0));
    return vec4(o, b);
}

// Escape disc: a spinning neon platter parked beside the pyramid.
vec4 drawDisc(vec2 q, float tb, float idx) {
    if (abs(q.x) > 0.055 || abs(q.y) > 0.030) return vec4(0.0);
    vec3 o = vec3(0.0);
    float cov = 0.0;
    float d = ell(q, vec2(0.0, 0.0), vec2(0.042, 0.014), 0.005);
    vec3 c = mix(vec3(0.30, 0.95, 1.00), palette(iHue + 0.35), 0.4);
    paint(o, cov, c * 0.7, d);
    // Chasing rim lights ride the rim, which can sit just outside the platter.
    for (int i = 0; i < 6; i++) {
        float fi = float(i);
        float a = fi / 6.0 * TAU + tb * 2.4 + idx * 1.7;
        vec2 lp = vec2(cos(a) * 0.036, sin(a) * 0.011);
        float m = smoothstep(0.007, 0.0, length(q - lp));
        paint(o, cov, c * (0.6 + iMid * 0.6) + c * 0.5, m);
    }
    o += c * d * 0.35;
    return vec4(o, cov);
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    vec2 uv = v_uv;
    vec2 p = (uv - 0.5) * vec2(A, 1.0);
    float t = iTime;
    float tb = mod(t, 600.0);

    vec3 col = backdrop(uv, A, tb);

    // Cubes, apex first so lower rows overlap correctly.
    for (int i = 0; i < 36; i++) {
        if (i >= iCubeCount) break;
        vec4 cu = iCube[i];
        vec2 cq = p - cu.xy;
        if (abs(cq.x) > iCubeHW * 1.15 || cq.y < -iCubeHW * 1.25 || cq.y > iCubeHW * 0.75)
            continue;
        vec3 cubeCol = mix(iColA, iColB, cu.z);
        // Pulse wave travelling down from the apex on every beat.
        float front = (1.0 - iPulse) * (iRows + 2.0);
        float wave = exp(-pow(cu.w - front, 2.0) * 1.1) * iPulse;
        float glow = 0.30 * (0.5 + 0.5 * sin(tb * 2.0 + cu.w * 0.5))
                   + iBass * 0.30 + cu.z * 0.25 + wave * 1.1;
        // The wave rides in through `glow` so it lights the cube's actual
        // faces. Adding it separately here would paint the cube's rectangular
        // bounding box instead, banding the whole row.
        col += drawCube(cq, cubeCol, glow, tb, cu.w * 7.0 + 1.0, iCubeHW);
    }

    // Characters composite at full opacity over the pyramid — `col` is
    // replaced where the sprite covers, so a cube never shows through a
    // character. Painter's order: props first, cast last.
    float s = iSprite;
    vec4 spr;
    for (int i = 0; i < 2; i++) {
        if (iDisc[i].z < 0.5) continue;
        spr = drawDisc((p - iDisc[i].xy) / s, tb, float(i));
        col = mix(col, spr.rgb, spr.a);
    }
    for (int i = 0; i < 4; i++) {
        if (iBall[i].z < 0.5) continue;
        spr = drawBall((p - iBall[i].xy) / s, tb);
        col = mix(col, spr.rgb, spr.a);
    }
    for (int i = 0; i < 2; i++) {
        if (iSlick[i].z < 0.5) continue;
        spr = drawSlick((p - iSlick[i].xy) / s, tb);
        col = mix(col, spr.rgb, spr.a);
    }

    if (iCoilyAlive > 0.5) {
        spr = drawCoily((p - iCoilyPos) / s, iCoilyEgg, iCoilyHop, tb);
        col = mix(col, spr.rgb * (1.0 + iBeat * 0.3), spr.a);
    }

    // Death blink drops coverage as well as colour, so the hero flickers out
    // rather than turning into a dark silhouette.
    float heroVis = (iHeroHit > 0.5) ? step(0.5, fract(tb * 14.0)) : 1.0;
    spr = drawQbert((p - iHeroPos) / s, sign(iHeroFace + 0.001), iHeroHop, tb);
    col = mix(col, spr.rgb * (1.0 + iMid * 0.2), spr.a * heroVis);
    // Riding a disc throws off a bright escape halo.
    col += mix(vec3(0.4, 1.0, 1.0), palette(iHue + 0.3), 0.4) * iHeroRide
         * smoothstep(0.14, 0.0, length(p - iHeroPos - vec2(0.0, 0.04))) * 0.9;

    col += palette(iHue + 0.5) * iFlash * 0.22;
    float sp = hash21(floor(p * iResolution.y * 0.5) + floor(tb * 20.0));
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

    _HOP_TIME = 0.26
    _PULSE_LIFT = 0.030
    # Course last shown by the previous instance, so a fresh activation never
    # opens on the same course twice running (same reasoning as Missile
    # Command's cityscape anti-repeat).
    _last_course: int | None = None

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
        self._pulse = 0.0
        self._hue = float(self.rng.uniform(0.0, 1.0))

        choices = [c for c in range(len(_COURSES)) if c != QBert._last_course]
        self._course = int(self.rng.choice(choices))
        QBert._last_course = self._course
        self._build_course()

    # ------------------------------------------------------------- geometry
    def _build_course(self) -> None:
        spec = _COURSES[self._course]
        self._rows = int(spec['rows'])
        self._mode = str(spec['mode'])
        self._col_a = tuple(spec['a'])
        self._col_b = tuple(spec['b'])

        # Per-row-count lattice, sized so every course fills a similar share of
        # the screen. Cubes are exactly 2*hw wide, so hw = dx/2 makes the
        # pyramid read as one solid stepped solid rather than floating tiles.
        dx, dy = _GEOM[self._rows]
        self._cube_hw = dx * 0.50
        self._sprite = self._cube_hw / 0.058
        # Centre the *occupied* band, which runs from the top of whoever is
        # standing on the apex down to the bottom skirt of the last row — not
        # just the cube geometry, or the apex character clips off the top.
        head = _CHAR_H * self._cube_hw
        skirt = _CUBE_DEPTH * self._cube_hw
        top_y = ((self._rows - 1) * dy + skirt - head) * 0.5

        self._cubes: list[tuple[int, int]] = []
        self._cube_pos: dict[tuple[int, int], tuple[float, float]] = {}
        for row in range(self._rows):
            for c in range(row + 1):
                self._cubes.append((row, c))
                self._cube_pos[(row, c)] = ((c - row / 2.0) * dx, top_y - row * dy)
        self._cube_state = {rc: 0.0 for rc in self._cubes}

        self._hero_cell = (0, 0)
        self._hero_from = self._hero_to = self._cube_pos[self._hero_cell]
        self._hero_hop_t = 1.0
        self._hero_hit = 0.0
        self._hero_face = 1.0
        self._hero_ride = 0.0

        self._coily_cell: tuple[int, int] | None = None
        self._coily_from = self._coily_to = (0.0, 0.0)
        self._coily_hop_t = 1.0
        self._coily_egg = 0.0
        self._coily_cd = float(self.rng.uniform(3.0, 6.0))
        self._coily_move_cd = 0.0

        self._balls: list[dict] = []
        self._ball_cd = float(self.rng.uniform(2.0, 5.0))
        self._slicks: list[dict] = []
        self._slick_cd = float(self.rng.uniform(8.0, 16.0))

        # Escape discs sit just outside the pyramid, two rows down.
        disc_row = min(2, self._rows - 1)
        lx, ly = self._cube_pos[(disc_row, 0)]
        rx, ry = self._cube_pos[(disc_row, disc_row)]
        off = self._cube_hw * 1.9
        self._discs = [{'pos': (lx - off, ly + self._cube_hw * 0.35), 'alive': True},
                       {'pos': (rx + off, ry + self._cube_hw * 0.35), 'alive': True}]

    def _next_course(self) -> None:
        choices = [c for c in range(len(_COURSES)) if c != self._course]
        self._course = int(self.rng.choice(choices))
        QBert._last_course = self._course
        self._hue = float((self._hue + self.rng.uniform(0.1, 0.25)) % 1.0)
        self._build_course()

    def _pulse_lift(self, row: int) -> float:
        """Vertical offset of a row under the current beat wave.

        Applied to cubes *and* to anything standing on them, so characters ride
        the wave with their platform instead of detaching from it.
        """
        if self._pulse <= 0.0:
            return 0.0
        front = (1.0 - self._pulse) * (self._rows + 2.0)
        return float(np.exp(-((row - front) ** 2) * 1.1) * self._pulse * self._PULSE_LIFT)

    # ------------------------------------------------------------------ AI
    def _neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        row, c = cell
        cands = [(row + 1, c), (row + 1, c + 1), (row - 1, c - 1), (row - 1, c)]
        return [(r, cc) for r, cc in cands if 0 <= r < self._rows and 0 <= cc <= r]

    def _needs_visit(self, rc: tuple[int, int]) -> bool:
        return self._cube_state[rc] < 1.0

    def _danger_cells(self) -> set[tuple[int, int]]:
        d: set[tuple[int, int]] = set()
        if self._coily_cell is not None and self._coily_egg <= 0.0:
            d.add(self._coily_cell)
            d.update(self._neighbors(self._coily_cell))
        for b in self._balls:
            d.add(b['cell'])
        return d

    def _arc_pos(self, frm, to, t: float, lift: float) -> tuple[float, float]:
        t = min(1.0, max(0.0, t))
        x = frm[0] + (to[0] - frm[0]) * t
        y = (frm[1] + (to[1] - frm[1]) * t
             + self._cube_hw * 0.75 * float(np.sin(np.pi * t)) + lift)
        return (x, y)

    def _hero_pos(self) -> tuple[float, float]:
        return self._arc_pos(self._hero_from, self._hero_to, self._hero_hop_t,
                             self._pulse_lift(self._hero_cell[0]))

    def _land_cube(self, rc: tuple[int, int]) -> None:
        """Apply this course's colouring rule to a cube Q*bert just landed on."""
        cur = self._cube_state[rc]
        if self._mode == 'once':
            self._cube_state[rc] = 1.0
        elif self._mode == 'two':
            self._cube_state[rc] = min(1.0, cur + 0.5)
        else:                                   # toggle
            self._cube_state[rc] = 0.0 if cur >= 1.0 else 1.0
        self._flash = max(self._flash, 0.15)

    def _start_hero_hop(self, dest: tuple[int, int]) -> None:
        self._hero_from = self._cube_pos[self._hero_cell]
        self._hero_to = self._cube_pos[dest]
        if dest[1] != self._hero_cell[1] or dest[0] < self._hero_cell[0]:
            self._hero_face = 1.0 if self._hero_to[0] >= self._hero_from[0] else -1.0
        self._hero_cell = dest
        self._hero_hop_t = 0.0

    def _try_disc_escape(self) -> bool:
        """Leap to a disc when Coily is closing in — Coily follows and dies."""
        if self._coily_cell is None or self._coily_egg > 0.0:
            return False
        if self._coily_cell not in self._neighbors(self._hero_cell) \
                and self._coily_cell != self._hero_cell:
            return False
        avail = [d for d in self._discs if d['alive']]
        if not avail:
            return False
        hx = self._cube_pos[self._hero_cell][0]
        disc = min(avail, key=lambda d: abs(d['pos'][0] - hx))
        disc['alive'] = False
        self._hero_from = self._cube_pos[self._hero_cell]
        self._hero_to = disc['pos']
        self._hero_hop_t = 0.0
        self._hero_ride = 1.0
        self._hero_face = 1.0 if disc['pos'][0] >= hx else -1.0
        self._coily_cell = None
        self._coily_egg = 0.0
        self._coily_cd = float(self.rng.uniform(4.0, 8.0))
        self._flash = max(self._flash, 0.9)
        return True

    def _update_hero(self, dt: float, tempo: float) -> None:
        if self._hero_ride > 0.0:
            self._hero_hop_t = min(1.0, self._hero_hop_t + dt / self._HOP_TIME * tempo)
            self._hero_ride = max(0.0, self._hero_ride - dt * 0.8)
            if self._hero_ride <= 0.0:
                # Ride over: back to the apex.
                self._hero_cell = (0, 0)
                self._hero_from = self._hero_to = self._cube_pos[self._hero_cell]
                self._hero_hop_t = 1.0
            return

        if self._hero_hop_t < 1.0:
            self._hero_hop_t += dt / self._HOP_TIME * tempo
            if self._hero_hop_t >= 1.0:
                self._land_cube(self._hero_cell)
            return

        if all(v >= 1.0 for v in self._cube_state.values()):
            self._flash = max(self._flash, 0.9)
            self._next_course()
            return

        if self._try_disc_escape():
            return

        danger = self._danger_cells()
        opts = self._neighbors(self._hero_cell)
        if not opts:
            return
        safe = [o for o in opts if o not in danger] or opts
        targets = [rc for rc in self._cubes if self._needs_visit(rc)]
        if not targets:
            return
        hc = self._hero_cell
        target = min(targets, key=lambda rc: (rc[0] - hc[0]) ** 2 + (rc[1] - hc[1]) ** 2)

        def cost(o: tuple[int, int]) -> tuple[int, int]:
            # In toggle courses, stepping back onto a finished cube undoes it.
            undo = 1 if (self._mode == 'toggle' and self._cube_state[o] >= 1.0) else 0
            return (undo, (o[0] - target[0]) ** 2 + (o[1] - target[1]) ** 2)

        self._start_hero_hop(min(safe, key=cost))

    def _kill_hero(self) -> None:
        self._hero_hit = 1.0
        self._flash = max(self._flash, 0.6)
        self._hero_cell = (0, 0)
        self._hero_from = self._hero_to = self._cube_pos[self._hero_cell]
        self._hero_hop_t = 1.0
        self._hero_ride = 0.0
        self._coily_cell = None
        self._coily_egg = 0.0
        self._coily_cd = float(self.rng.uniform(3.0, 6.0))

    def _update_coily(self, dt: float, tempo: float) -> None:
        if self._coily_cell is None:
            self._coily_cd -= dt
            if self._coily_cd <= 0.0:
                self._coily_cell = (0, 0)
                self._coily_from = self._coily_to = self._cube_pos[self._coily_cell]
                self._coily_hop_t = 1.0
                self._coily_move_cd = 0.0
                self._coily_egg = 1.0        # arrives as an egg, hatches below
            return

        if self._coily_hop_t < 1.0:
            self._coily_hop_t += dt / (self._HOP_TIME * 1.25) * tempo
            return

        self._coily_move_cd -= dt
        if self._coily_move_cd <= 0.0:
            opts = self._neighbors(self._coily_cell)
            if opts:
                if self._coily_egg > 0.0:
                    # Egg bounces blindly downward, then hatches.
                    down = [o for o in opts if o[0] > self._coily_cell[0]]
                    nxt = (down[int(self.rng.integers(0, len(down)))] if down
                           else opts[int(self.rng.integers(0, len(opts)))])
                    if self._coily_cell[0] >= min(3, self._rows - 1):
                        self._coily_egg = 0.0
                        self._flash = max(self._flash, 0.35)
                else:
                    hc = self._hero_cell
                    nxt = min(opts, key=lambda o: (o[0] - hc[0]) ** 2 + (o[1] - hc[1]) ** 2)
                self._coily_from = self._cube_pos[self._coily_cell]
                self._coily_to = self._cube_pos[nxt]
                self._coily_cell = nxt
                self._coily_hop_t = 0.0
            self._coily_move_cd = float(self.rng.uniform(0.35, 0.7)) / (1.0 + self._mid * 0.5)

        if (self._coily_egg <= 0.0 and self._coily_cell == self._hero_cell
                and self._hero_hit <= 0.0 and self._hero_ride <= 0.0):
            self._kill_hero()

    def _update_balls(self, dt: float, tempo: float) -> None:
        self._ball_cd -= dt
        if self._ball_cd <= 0.0 and len(self._balls) < _NBALL and self._rows > 2:
            start_c = int(self.rng.integers(0, 2))
            self._balls.append({
                'cell': (1, start_c),
                'from': self._cube_pos[(1, start_c)],
                'to': self._cube_pos[(1, start_c)],
                'hop': 1.0,
                'cd': 0.0,
            })
            self._ball_cd = float(self.rng.uniform(2.5, 6.0)) / (1.0 + self._bass * 0.4)

        for b in list(self._balls):
            if b['hop'] < 1.0:
                b['hop'] += dt / (self._HOP_TIME * 1.1) * tempo
                continue
            b['cd'] -= dt
            if b['cd'] > 0.0:
                continue
            row, c = b['cell']
            if row + 1 >= self._rows:
                self._balls.remove(b)
                continue
            nxt = (row + 1, c + int(self.rng.integers(0, 2)))
            b['from'] = self._cube_pos[b['cell']]
            b['to'] = self._cube_pos[nxt]
            b['cell'] = nxt
            b['hop'] = 0.0
            b['cd'] = float(self.rng.uniform(0.25, 0.5))
            if nxt == self._hero_cell and self._hero_hit <= 0.0 and self._hero_ride <= 0.0:
                self._kill_hero()

    def _update_slicks(self, dt: float, tempo: float) -> None:
        self._slick_cd -= dt
        if self._slick_cd <= 0.0 and len(self._slicks) < _NSLICK and self._rows > 2:
            start_c = int(self.rng.integers(0, 2))
            self._slicks.append({
                'cell': (1, start_c),
                'from': self._cube_pos[(1, start_c)],
                'to': self._cube_pos[(1, start_c)],
                'hop': 1.0,
                'cd': 0.0,
            })
            self._slick_cd = float(self.rng.uniform(9.0, 18.0))

        for s in list(self._slicks):
            if s['hop'] < 1.0:
                s['hop'] += dt / (self._HOP_TIME * 1.35) * tempo
                if s['hop'] >= 1.0:
                    # Steals the colour back off whatever it lands on.
                    if self._cube_state[s['cell']] > 0.0:
                        self._cube_state[s['cell']] = 0.0
                        self._flash = max(self._flash, 0.25)
                continue
            s['cd'] -= dt
            if s['cd'] > 0.0:
                continue
            row, c = s['cell']
            if row + 1 >= self._rows:
                self._slicks.remove(s)
                continue
            nxt = (row + 1, c + int(self.rng.integers(0, 2)))
            s['from'] = self._cube_pos[s['cell']]
            s['to'] = self._cube_pos[nxt]
            s['cell'] = nxt
            s['hop'] = 0.0
            s['cd'] = float(self.rng.uniform(0.4, 0.8))

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
            self._pulse = 1.0
            self._flash = max(self._flash, 0.25)
        self._beat = max(0.0, self._beat - dt * 2.5)
        self._pulse = max(0.0, self._pulse - dt * 1.5)
        self._flash = max(0.0, self._flash - dt * 1.8)
        self._hero_hit = max(0.0, self._hero_hit - dt * 1.4)

        tempo = max(0.05, float(self.parameters['speed'])) * (1.0 + self._bass * 0.5)
        self._update_hero(dt, tempo)
        self._update_coily(dt, tempo)
        self._update_balls(dt, tempo)
        self._update_slicks(dt, tempo)

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
        p['iPulse'].value = float(self._pulse)
        p['iRows'].value = float(self._rows)
        p['iCubeHW'].value = float(self._cube_hw)
        p['iSprite'].value = float(self._sprite)
        p['iColA'].value = self._col_a
        p['iColB'].value = self._col_b

        cube_val = []
        for rc in self._cubes:
            x, y = self._cube_pos[rc]
            cube_val.append((x, y + self._pulse_lift(rc[0]),
                             float(self._cube_state[rc]), float(rc[0])))
        while len(cube_val) < _NCUBE_MAX:
            cube_val.append((_OFF, _OFF, 0.0, 0.0))
        p['iCube'].value = cube_val
        p['iCubeCount'].value = len(self._cubes)

        p['iHeroPos'].value = self._hero_pos()
        p['iHeroFace'].value = float(self._hero_face)
        p['iHeroHop'].value = float(min(1.0, self._hero_hop_t))
        p['iHeroHit'].value = 1.0 if self._hero_hit > 0.0 else 0.0
        p['iHeroRide'].value = float(self._hero_ride)

        if self._coily_cell is not None:
            p['iCoilyPos'].value = self._arc_pos(
                self._coily_from, self._coily_to, self._coily_hop_t,
                self._pulse_lift(self._coily_cell[0]))
            p['iCoilyHop'].value = float(min(1.0, self._coily_hop_t))
            p['iCoilyAlive'].value = 1.0
            p['iCoilyEgg'].value = float(self._coily_egg)
        else:
            p['iCoilyPos'].value = (_OFF, _OFF)
            p['iCoilyHop'].value = 0.0
            p['iCoilyAlive'].value = 0.0
            p['iCoilyEgg'].value = 0.0

        ball_val = []
        for i in range(_NBALL):
            if i < len(self._balls):
                b = self._balls[i]
                bx, by = self._arc_pos(b['from'], b['to'], b['hop'],
                                       self._pulse_lift(b['cell'][0]))
                ball_val.append((bx, by, 1.0))
            else:
                ball_val.append((_OFF, _OFF, 0.0))
        p['iBall'].value = ball_val

        slick_val = []
        for i in range(_NSLICK):
            if i < len(self._slicks):
                s = self._slicks[i]
                sx, sy = self._arc_pos(s['from'], s['to'], s['hop'],
                                       self._pulse_lift(s['cell'][0]))
                slick_val.append((sx, sy, 1.0))
            else:
                slick_val.append((_OFF, _OFF, 0.0))
        p['iSlick'].value = slick_val

        p['iDisc'].value = [
            (d['pos'][0], d['pos'][1], 1.0) if d['alive'] else (_OFF, _OFF, 0.0)
            for d in self._discs
        ]

        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()

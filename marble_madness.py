"""Marble Madness — an isometric chute, and a marble that rolls down it.

The course is a height field, `courseY(x, z)`, and it is the only description
of the course that exists. The renderer finds the surface by walking a ray
into that function; the simulation accelerates the marble down its gradient.
Neither owns a mesh, so there is nothing for them to disagree about: if the
chute banks, the marble banks, because banking *is* a gradient.

The marble is not animated along a path. It is given gravity resolved along
the surface -- a solid sphere rolling without slipping accelerates at 5/7 g
sin(theta), not g sin(theta), and the difference is the third of the energy
that goes into spin instead of speed -- and it goes where that takes it. The
banked rim is what turns it back, so a fast marble climbs high on the outside
of a bend and a slow one hugs the inside, which is exactly the behaviour the
arcade cabinet sold.

Rendering an orthographic height field naively costs about 85 samples per
pixel. It costs 24 here, because the course is a tilted plane plus a bounded
correction: the ray's intersection with the plane is solved in closed form,
the correction is known to lie within [-6, +7] world units, and only that
window is ever searched. See `traceCourse`.

Audio reactivity:
  - bass    -> the shove down the chute, and the marble's own glow
  - mid     -> palette drift over the course tiles
  - treble  -> sparkle on the rim and the marble's specular
  - beat    -> the marble hops, and lands with a ring of dust

Randomised per activation: palette, the chute's snake, the start position,
where the rivals are, and which way the marble is already drifting.

Config keys (under ``[effects.MarbleMadness]``):
    speed, reactivity, glitter
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.17.0'

_SLOPE = 0.22           # how steeply the chute descends along +x
_HALF = 13.0            # chute half-width before the edge gives way
_RIM_IN = 4.5           # how far in from the edge the banking starts
_RIM_H = 9.0            # how high the banked rim climbs
_DROP = 17.0            # how far the world falls away past the edge
_RADIUS = 1.70
_G = 26.0
_DAMP = 0.75            # rolling resistance, as an exponential decay rate
# The last fraction of a unit before the edge is a wall. The banked rim does
# most of the containing, but it cannot do all of it: past the edge the
# surface falls away at a slope of about 21, roughly five times what the
# steepest part of the banking pulls back, so crossing the rim even once is
# terminal. Measured over 60 s without it, the marble spent 43% of its frames
# off the course and drifted 46 units from a centreline it was meant to hold
# within 13.
_WALL = _HALF - 0.45
_BOUNCE = 0.55
_NRIVAL = 3
_TAU = 2.0 * math.pi

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
uniform float iPunch;      // the beat's transient, decaying ~3x faster
uniform vec4  iBand[4];    // 16 FFT bands, low to high
uniform float iHue;
uniform float iGlitter;
uniform float iZoom;
uniform float iLand;
uniform vec3  iSnake;      // centreline phases folded to [0, 2pi), then cam z
uniform mat3  iRoll;       // the player marble's accumulated roll
uniform vec4  iBall[4];    // camera-relative x, y, z, radius; [0] is the player
in  vec2 v_uv;
out vec4 fragColor;

#define TAU 6.28318530
// The camera basis. Fixed: this is an isometric cabinet, the view never turns.
const vec3 FWD = vec3(0.5773503, -0.5773503, -0.5773503);
const vec3 RGT = vec3(0.7071068, 0.0, 0.7071068);
const vec3 UPV = vec3(0.4082483, 0.8164966, -0.4082483);

float hash21(vec2 p) {
    p = mod(p, 512.0);
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

// The 16 bands, packed four to a vec4. `bandU` reads the spectrum as a curve
// across 0..1, so the chute's cross-section can be driven straight off it.
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


// The chute's centreline. Camera-relative x, with the absolute phase carried
// in as a uniform so nothing here ever sees a large coordinate.
float centre(float x) {
    return 6.0 * sin(x * 0.040 + iSnake.x)
         + 3.2 * sin(x * 0.097 + iSnake.y) - iSnake.z;
}

// How far the surface sits above the chute's mean plane. Bounded by
// construction to [_RIM_H - _DROP, _RIM_H] = [-8, +9], which is what makes
// the short search window in traceCourse safe.
float relief(float x, float z) {
    float d = abs(z - centre(x));
    float u = clamp((d - (13.0 - 4.5)) / 4.5, 0.0, 1.0);
    return u * u * 9.0 - 17.0 * smoothstep(13.0, 13.0 + 0.6, d);
}

float courseY(float x, float z) { return -x * 0.22 + relief(x, z); }

// Walk the ray into the height field.
//
// The naive version marches the whole depth range and costs ~85 samples. But
// the course is a tilted plane plus a correction that is bounded, so the
// plane crossing can be solved exactly and only the band the correction can
// reach needs searching. `k` is how fast the ray closes on the plane, so a
// relief of r moves the hit by r/k along the ray -- and r is bounded by
// construction.
bool traceCourse(vec3 o, out float tHit, out vec3 pos) {
    float k = -(FWD.y + FWD.x * 0.22);
    if (k < 1e-4) return false;
    float t0 = -(o.x * 0.22 + o.y) / (FWD.y + FWD.x * 0.22);
    float tA = t0 - 9.0 / k;
    float tB = t0 + (17.0 - 9.0 + 1.0) / k;
    float prev = tA;
    const int N = 24;
    for (int i = 1; i <= N; i++) {
        float t = tA + (tB - tA) * float(i) / float(N);
        vec3 p = o + FWD * t;
        float g = p.y - courseY(p.x, p.z);
        if (g < 0.0) {
            // Bisect between the last sample above the surface and this one.
            float lo = prev, hi = t;
            for (int j = 0; j < 5; j++) {
                float m = 0.5 * (lo + hi);
                vec3 q = o + FWD * m;
                if (q.y - courseY(q.x, q.z) < 0.0) hi = m; else lo = m;
            }
            tHit = 0.5 * (lo + hi);
            pos = o + FWD * tHit;
            return true;
        }
        prev = t;
    }
    return false;
}

vec3 courseNormal(vec3 p) {
    float e = 0.30;
    float dx = courseY(p.x + e, p.z) - courseY(p.x - e, p.z);
    float dz = courseY(p.x, p.z + e) - courseY(p.x, p.z - e);
    return normalize(vec3(-dx, 2.0 * e, -dz));
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float t = mod(iTime, 600.0);
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0);
    vec3 neon = palette(iHue);

    // Orthographic: every pixel is its own ray origin, one shared direction.
    vec3 o = RGT * (p.x * iZoom) + UPV * (p.y * iZoom) - FWD * 120.0;

    vec3 col;
    float tC;
    vec3 hit;
    bool got = traceCourse(o, tC, hit);

    // The void the course hangs in.
    float sky = clamp(p.y * 0.9 + 0.5, 0.0, 1.0);
    vec3 bg = mix(vec3(0.015, 0.020, 0.055), vec3(0.05, 0.03, 0.12), sky);
    bg += palette(iHue + 0.5) * 0.055 * (1.0 - sky);
    // Dust in the void, as round points inside their cells rather than a
    // threshold on the cell hash, which would fill whole squares.
    vec2 cell = (p + vec2(t * 0.004, 0.0)) * 46.0;
    vec2 cf = fract(cell) - 0.5;
    bg += vec3(0.7, 0.8, 1.0) * step(0.985, hash21(floor(cell)))
        * smoothstep(0.34, 0.02, length(cf))
        * (0.30 + iTreble * 1.5 + iPunch * 1.3) * iGlitter;
    col = bg;

    if (got) {
        vec3 n = courseNormal(hit);
        float d = abs(hit.z - centre(hit.x));
        // Tiles: the classic chequered ramp, laid out along the chute and
        // across it so the pattern turns with the bends instead of sliding.
        float chk = mod(floor(hit.x * 0.34) + floor(d * 0.34), 2.0);
        vec3 tileA = mix(vec3(0.62, 0.60, 0.72), palette(iHue + 0.10), 0.45);
        vec3 tileB = mix(vec3(0.30, 0.29, 0.40), palette(iHue + 0.55), 0.30);
        vec3 base = mix(tileA, tileB, chk);
        base = mix(base, palette(iHue + 0.3), iMid * 0.18);
        // The chute's cross-section is the spectrum: low bands down the
        // centreline, high bands out at the banking. The tiles were the
        // largest surface in frame and took nothing from the music.
        float bu = clamp(d / 13.0, 0.0, 1.0);
        float lb = bandU(bu);
        // Modulating the tiles, not replacing them: at 0.55/0.32 the bands
        // read as painted stripes and the chequer -- which is what says
        // "Marble Madness" -- stopped being the dominant pattern.
        base *= 1.0 + lb * 0.30;
        base += mix(palette(iHue + 0.15), vec3(1.0), 0.30) * lb * 0.16;
        // The banked rim reads as a separate material, lit from within.
        float rim = smoothstep(13.0 - 4.5, 13.0 - 0.3, d);
        base = mix(base, mix(neon, vec3(1.0), 0.25), rim * 0.30);

        vec3 L = normalize(vec3(0.45, 0.82, 0.36));
        float dif = clamp(dot(n, L), 0.0, 1.0);
        float amb = 0.30 + 0.20 * n.y;
        col = base * (amb + dif * 0.85);
        // Past the edge the surface continues as a lower plain. Keeping that
        // geometry is what gives the chute a shaded side wall and therefore
        // any thickness at all -- but shading it like more course tiles the
        // whole screen and the course stops reading as a course. So it stays
        // geometry and becomes abyss: untiled, barely lit, fogged out.
        float out_ = smoothstep(13.0 - 0.15, 13.0 + 0.55, d);
        vec3 abyss = mix(vec3(0.020, 0.024, 0.062), palette(iHue + 0.5), 0.07);
        col = mix(col, abyss * (0.26 + dif * 0.34), out_);
        // Everything below is course lighting, so it is masked out of the
        // abyss. `rim` is 1 everywhere past the edge, so leaving it unmasked
        // washes the entire lower plain in rim glow.
        float on = 1.0 - out_;
        col += neon * rim * (0.04 + iBass * 0.55 + lb * 0.50 + iPunch * 0.45) * on;
        // A lip of light exactly on the edge, which is where the eye reads the
        // shape of the course from. Narrow: widened into the banking it stops
        // being an edge and becomes a wash over most of the chute.
        col += mix(neon, vec3(1.0), 0.4)
             * smoothstep(0.40, 0.0, abs(d - 13.0))
             * (0.26 + iTreble * 0.9 + iPunch * 1.1);
        // The ring thrown out where the marble came back down.
        if (iLand > 0.0) {
            float rr = length(hit.xz - iBall[0].xz);
            col += mix(neon, vec3(1.0), 0.35)
                 * smoothstep(1.1, 0.0, abs(rr - (1.0 - iLand) * 8.0))
                 * iLand * 0.85 * on;
        }

        // Glitter lying on the tiles, in world space so it stays put as the
        // camera runs over it. Round points inside their cells -- a bare
        // threshold on the cell hash would light whole squares.
        vec2 gc = vec2(hit.x, hit.z) * 1.15;
        vec2 gf = fract(gc) - 0.5;
        col += mix(vec3(1.0), neon, 0.35) * on
             * step(0.955 - iPunch * 0.030, hash21(floor(gc)))
             * smoothstep(0.34, 0.02, length(gf))
             * (0.20 + iTreble * 1.7 + iPunch * 1.5) * iGlitter;
        // Contact shadows under each marble.
        for (int i = 0; i < 4; i++) {
            vec4 B = iBall[i];
            if (B.w <= 0.0) continue;
            vec2 dv = hit.xz - B.xz;
            float sh = smoothstep(B.w * 1.9, B.w * 0.35, length(dv));
            col *= 1.0 - sh * 0.55;
        }
        // Distance haze, so the chute recedes instead of stacking up flat.
        col = mix(col, bg, smoothstep(0.35, 1.0, abs(hit.x) / 62.0));
        col = mix(col, bg, out_ * smoothstep(6.0, 46.0, abs(hit.x)) * 0.85);
    }

    // ---- the marbles ----
    // Orthographic ray/sphere: the closest approach is a dot product, so this
    // is a couple of instructions and it composites against the course by
    // depth rather than by paint order.
    for (int i = 0; i < 4; i++) {
        vec4 B = iBall[i];
        if (B.w <= 0.0) continue;
        vec3 oc = B.xyz - o;
        float tca = dot(oc, FWD);
        float d2 = dot(oc, oc) - tca * tca;
        float r2 = B.w * B.w;
        if (d2 > r2) continue;
        float tS = tca - sqrt(r2 - d2);
        if (got && tS > tC) continue;
        vec3 sp = o + FWD * tS;
        vec3 n = (sp - B.xyz) / B.w;
        vec3 L = normalize(vec3(0.45, 0.82, 0.36));
        float dif = clamp(dot(n, L), 0.0, 1.0);
        float fres = pow(1.0 - clamp(dot(n, -FWD), 0.0, 1.0), 3.0);
        vec3 body;
        if (i == 0) {
            // The player's marble carries its own roll, so the banding turns
            // with the surface it is rolling on -- a sphere with a static
            // texture reads as a sliding decal, not as something rolling.
            vec3 s = iRoll * n;
            float band = 0.5 + 0.5 * sin(atan(s.z, s.x) * 3.0 + s.y * 2.2);
            body = mix(vec3(0.93, 0.94, 0.98), mix(neon, vec3(1.0), 0.2), band * 0.55);
            body *= 0.85 + 0.30 * step(0.0, sin(s.y * 9.0));
        } else {
            body = vec3(0.08, 0.07, 0.10);
            vec3 s = iRoll * n;
            body += vec3(0.16, 0.05, 0.20) * step(0.0, sin(s.x * 7.0 + s.y * 4.0));
        }
        vec3 c = body * (0.22 + dif * 0.95);
        c += vec3(1.0) * pow(clamp(dot(reflect(-L, n), -FWD), 0.0, 1.0), 42.0)
           * (0.6 + iTreble * 1.6 + iPunch * 1.4);
        c += neon * fres * (0.30 + iBass * 1.5 + iPunch * 1.1)
           * (i == 0 ? 1.0 : 0.45);
        col = c;
    }

    col += neon * (iBeat * 0.06 + iPunch * 0.10);
    col *= smoothstep(1.30, 0.34, length(p));
    col = col / (1.0 + col * 0.32);
    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.88)), 1.0);
}
"""


class MarbleMadness(BaseEffect):
    """Isometric chute rendered from a height field, with a marble rolling it."""

    NAME = 'Marble Madness'
    AUTHOR = 'unicorn-viz'
    TAGS = ['games', 'arcade', 'neon', 'energetic', 'groovy']
    PING_PONG_FRIENDS = [
        'Q*bert', 'Super Mario', 'Mario Kart', 'Frogger', 'Breakout',
        'Donkey Kong',
    ]

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
        # ever moves at the centre of the chute and the banking looks dead.
        self._band_tilt = np.linspace(1.0, 2.7, 16)
        self._tbl_band = np.zeros((4, 4), dtype='f4')
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._glitter = max(0.0, float(self.parameters['glitter']))
        # World units across the short screen axis. The chute is 26 wide, so
        # anything under about 30 fills the frame with tiles and the course
        # stops reading as a course hanging in a void.
        self._zoom = 34.0

        # The chute's shape. Phases only -- the amplitudes are fixed so the
        # relief stays inside the bounds traceCourse relies on.
        self._ph1 = float(self.rng.uniform(0.0, _TAU))
        self._ph2 = float(self.rng.uniform(0.0, _TAU))

        self._x = float(self.rng.uniform(0.0, 400.0))
        self._z = self._centre(self._x) + float(self.rng.uniform(-4.0, 4.0))
        self._vx = 6.0
        self._vz = float(self.rng.uniform(-3.0, 3.0))
        self._y = self._course_y(self._x, self._z) + _RADIUS
        self._vy = 0.0
        self._air = False
        self._land = 0.0
        self._roll = np.eye(3, dtype=np.float64)

        # Rivals run the same physics from their own starting points.
        self._riv = []
        for i in range(_NRIVAL):
            rx = self._x + float(self.rng.uniform(-26.0, 34.0))
            self._riv.append({
                'x': rx, 'z': self._centre(rx) + float(self.rng.uniform(-6.0, 6.0)),
                'vx': 6.0, 'vz': float(self.rng.uniform(-3.0, 3.0)),
                'r': float(self.rng.uniform(0.85, 1.10)),
            })

        self._tbl_ball = np.zeros((4, 4), dtype='f4')

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

    # -------------------------------------------------------------- the course
    def _centre(self, x: float) -> float:
        """The chute's centreline -- the shader's `centre()`, in absolute x."""
        return 6.0 * math.sin(x * 0.040) + 3.2 * math.sin(x * 0.097)

    def _course_y(self, x: float, z: float) -> float:
        """Surface height, matching the shader's `courseY()` exactly."""
        d = abs(z - self._centre(x))
        u = min(1.0, max(0.0, (d - (_HALF - _RIM_IN)) / _RIM_IN))
        e = (d - _HALF) / 0.6
        e = min(1.0, max(0.0, e))
        drop = _DROP * (e * e * (3.0 - 2.0 * e))
        return -x * _SLOPE + u * u * _RIM_H - drop

    def _grad(self, x: float, z: float) -> tuple[float, float]:
        """Central difference of the surface -- what the marble rolls down."""
        e = 0.30
        gx = (self._course_y(x + e, z) - self._course_y(x - e, z)) / (2.0 * e)
        gz = (self._course_y(x, z + e) - self._course_y(x, z - e)) / (2.0 * e)
        return gx, gz

    def _centre_slope(self, x: float) -> float:
        """d(centreline)/dx -- how fast the chute itself slides sideways."""
        return (6.0 * 0.040 * math.cos(x * 0.040)
                + 3.2 * 0.097 * math.cos(x * 0.097))

    def _step_ball(self, x: float, z: float, vx: float, vz: float,
                   dt: float, push: float) -> tuple[float, float, float, float]:
        """Roll one marble for a frame and keep it inside the chute.

        Gravity is resolved down the surface gradient, which is the whole of
        the steering: the banked rim turns the marble because banking *is* a
        gradient. The wall at the edge is the backstop described at _WALL.

        That wall is moving, because the chute is. What reflects is the
        marble's velocity *relative* to the centreline -- reflecting the world
        velocity instead would fight the chute's own drift and stall the
        marble against a bend.
        """
        gx, gz = self._grad(x, z)
        # A solid sphere rolling without slipping takes 5/7 g sin(theta): the
        # missing two sevenths go into spin, not into loss.
        k = 5.0 / 7.0 * _G
        vx += (-k * gx + push) * dt
        vz += -k * gz * dt
        damp = math.exp(-_DAMP * dt)
        vx *= damp
        vz *= damp
        x += vx * dt
        z += vz * dt

        c = self._centre(x)
        off = z - c
        if abs(off) > _WALL:
            side = math.copysign(1.0, off)
            z = c + side * _WALL
            drift = self._centre_slope(x) * vx
            rel = vz - drift
            if rel * side > 0.0:
                vz = drift - rel * _BOUNCE
        return x, z, vx, vz

    # ------------------------------------------------------------------- sim
    def update(self, dt: float, audio: AudioData) -> None:
        """Roll the marble down the gradient; everything else follows from it."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 1.8))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 1.8))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 1.8))
        self._pump_bands(audio, dt)
        beat_now = audio.beat > 0.5
        if beat_now:
            self._beat = 1.0
            self._punch = 1.0
        self._beat = max(0.0, self._beat - dt * 3.4)
        # A second, much faster envelope: iBeat at 3.4/s is a ~300 ms swell,
        # which reads as a pulse rather than as a hit.
        self._punch = max(0.0, self._punch - dt * 11.0)
        self._land = max(0.0, self._land - dt * 1.9)

        pace = max(0.05, float(self.parameters['speed']))
        push = (2.6 + self._bass * 8.0) * pace

        self._x, self._z, self._vx, self._vz = self._step_ball(
            self._x, self._z, self._vx, self._vz, dt, push)

        # A beat launches it; gravity brings it back to the surface.
        floor = self._course_y(self._x, self._z) + _RADIUS
        if beat_now and not self._air:
            self._vy = 7.0 + self._bass * 5.5
            self._air = True
        if self._air:
            self._vy -= _G * dt
            self._y += self._vy * dt
            if self._y <= floor:
                self._y = floor
                self._vy = 0.0
                self._air = False
                self._land = 1.0
        else:
            self._y = floor

        # Rolling without slipping: the contact point is stationary, so the
        # spin axis is the surface normal crossed with the velocity. Integrating
        # the rotation itself (rather than an angle) is what lets the banding
        # on the marble stay glued to the marble through a turn.
        speed = math.hypot(self._vx, self._vz)
        if speed > 1e-4:
            axis = np.array([-self._vz, 0.0, self._vx]) / speed
            ang = speed / _RADIUS * dt
            self._roll = _rodrigues(axis, ang) @ self._roll
            if not np.isfinite(self._roll).all():
                self._roll = np.eye(3)

        for b in self._riv:
            b['x'], b['z'], b['vx'], b['vz'] = self._step_ball(
                b['x'], b['z'], b['vx'], b['vz'], dt, push * 0.92)
            # Keep the field around the player rather than letting it string
            # out over a kilometre of chute.
            if b['x'] - self._x > 46.0:
                b['x'] = self._x - float(self.rng.uniform(16.0, 30.0))
                b['z'] = self._centre(b['x']) + float(self.rng.uniform(-6.0, 6.0))
            elif self._x - b['x'] > 40.0:
                b['x'] = self._x + float(self.rng.uniform(14.0, 32.0))
                b['z'] = self._centre(b['x']) + float(self.rng.uniform(-6.0, 6.0))

    def render(self) -> None:
        """Upload camera-relative geometry; the shader traces the rest.

        The shader works entirely in camera-relative coordinates, which is what
        keeps large world positions away from the trigonometry. That change of
        variables only closes if the camera's *height* is the chute's mean
        plane at the camera's x: the plane term is `-x * _SLOPE`, so shifting x
        by cx shifts the plane by `-cx * _SLOPE`, and the camera has to absorb
        exactly that. Centre the camera on the marble instead and the whole
        course renders at the wrong height.
        """
        cx, cz = self._x, self._z
        cy = -cx * _SLOPE
        self._tbl_ball[:] = 0.0
        self._tbl_ball[0] = (0.0, self._y - cy, 0.0, _RADIUS)
        for i, b in enumerate(self._riv[:3]):
            by = self._course_y(b['x'], b['z']) + b['r']
            self._tbl_ball[i + 1] = (b['x'] - cx, by - cy, b['z'] - cz, b['r'])

        p = self._prog
        p['iResolution'].value = (float(self.width), float(self.height))
        p['iTime'].value = float(self.time)
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iPunch'].value = self._punch
        p['iBand'].write(self._tbl_band.tobytes())
        p['iHue'].value = self._hue
        p['iGlitter'].value = self._glitter
        p['iZoom'].value = self._zoom
        p['iLand'].value = float(self._land)
        # The centreline's absolute phase travels as a folded offset, and its
        # z is carried alongside, so no large coordinate reaches the shader.
        p['iSnake'].value = (float((cx * 0.040) % _TAU),
                             float((cx * 0.097) % _TAU), float(cz))
        p['iRoll'].write(np.ascontiguousarray(self._roll.T, dtype='f4').tobytes())
        p['iBall'].write(self._tbl_ball.tobytes())
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()


def _rodrigues(axis: np.ndarray, ang: float) -> np.ndarray:
    """Rotation matrix for *ang* radians about a unit *axis*."""
    kx, ky, kz = axis
    kmat = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + math.sin(ang) * kmat + (1.0 - math.cos(ang)) * (kmat @ kmat)

"""Frogger — a frog that actually reads the traffic before it hops.

The point of Frogger is not the hopping, it is the *waiting*. A frog that
hops on a timer gets flattened; a frog that hops when the gap is there reads
as alive. So the simulation is built around a single question asked once per
hop: if I land on that tile, will anything be occupying it when I get there?
For the road that means sweeping every car's span forward over the hop's
duration, and for the river it means the opposite -- refusing to hop unless
something *will* be there to land on.

Everything downstream falls out of that. The frog rides a log because it is
standing on one and the log is moving. It drowns because it hopped at a log
that had already gone. It waits in the median for four lanes to line up, and
when they do it crosses in one run. None of those behaviours are scripted.

The renderer is the pack's usual split: the simulation runs on the CPU and
uploads flat arrays of spans, and the shader only rasterizes them. No geometry
is evaluated per pixel that was not already decided per frame.

Audio reactivity:
  - bass    -> traffic speed and the hop cadence
  - mid     -> palette drift, water colour
  - treble  -> water sparkle, headlight flare
  - beat    -> the frog takes its hop on the beat when one is due

Randomised per activation: palette hue, lane directions, speeds, vehicle mix,
log lengths, which bays start filled, and the frog's starting column.

Config keys (under ``[effects.Frogger]``):
    speed, reactivity, glitter
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.16.0'

_ROWS = 13              # 0 bank, 1-5 road, 6 median, 7-11 river, 12 homes
_ROAD_LO, _ROAD_HI = 1, 5
_MEDIAN = 6
_RIVER_LO, _RIVER_HI = 7, 11
_HOME_ROW = 12
_HALF_W = 13.0          # playfield half-width in tiles; wider than the screen
_NCAR = 20
_NLOG = 18
_NHOME = 5
_HOP_T = 0.26           # seconds in the air for one hop
# A tile has to be clear for the hop *and* for the time the frog then stands on
# it. Checking only the landing instant is what a naive Frogger AI does, and it
# dies constantly -- it steps into a real gap and is still standing there when
# the next car arrives. Measured over 90 s at 0.26 s of lookahead: 26 deaths and
# the frog never got past the third lane.
# Sized to the dwell the frog actually takes between decisions. Padding this
# beyond the truth does not buy safety, it just shrinks every window: at 0.55
# against a real dwell of ~0.38 the frog spent 40% of its life parked on the
# bank and the median waiting for gaps that were in fact already open.
_DWELL = 0.40
_LOOK = _HOP_T + _DWELL
# How close a threat may get before the frog stops waiting for the beat.
_PANIC = 0.42
_OFF = -999.0

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
uniform float iHue;
uniform float iGlitter;
uniform vec4  iCar[20];     // x, row, half-length, kind
uniform vec4  iLog[18];     // x, row, half-length, kind (<0 = submerged turtle)
uniform vec4  iHome[5];     // x, filled, glow, unused
uniform vec4  iFrog;        // x, y, hop height, state (0 alive, >0 dying)
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

// Rounded box, in tile units.
float sdBox(vec2 p, vec2 b, float r) {
    vec2 d = abs(p) - b + r;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - r;
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float t = mod(iTime, 600.0);
    // Tiles are square and the playfield is 13.0 tall, so the columns that
    // fall off the sides are exactly where traffic enters and leaves.
    vec2 g = vec2((v_uv.x - 0.5) * A * 13.0, v_uv.y * 13.0);
    float row = floor(g.y);
    vec3 neon = palette(iHue);
    vec3 col;

    // ---- the bands ----
    if (row >= 7.0 && row <= 11.0) {
        // Water: a swell built from two trains, one of them bent by the
        // other so the crests break up instead of running as straight bars.
        float w = 0.62 * sin(g.x * 2.4 + t * 1.2 + sin(g.y * 2.7 + t * 0.6) * 0.9)
                + 0.38 * sin(g.x * 5.3 - t * 2.0 + g.y * 1.7);
        col = mix(vec3(0.02, 0.08, 0.24), vec3(0.04, 0.19, 0.46), w * 0.5 + 0.5);
        col = mix(col, palette(iHue + 0.55), 0.09 + iMid * 0.07);
        col += vec3(0.45, 0.70, 0.95) * smoothstep(0.72, 1.0, w) * 0.16;
        // Glitter as round points inside their cells. Thresholding the cell
        // hash on its own lights the whole cell and the water fills up with
        // hard little squares.
        vec2 cell = vec2(g.x * 8.0, g.y * 8.0 - t * 1.4);
        vec2 cf = fract(cell) - 0.5;
        float sp = step(0.975, hash21(floor(cell)))
                 * smoothstep(0.34, 0.02, length(cf));
        col += vec3(0.70, 0.88, 1.0) * sp * (0.45 + iTreble * 0.9) * iGlitter;
    } else if (row >= 1.0 && row <= 5.0) {
        col = vec3(0.155, 0.155, 0.175) * (0.88 + 0.24 * hash21(floor(g * 6.0)));
        // Dashed lane markings on the seam between lanes.
        float dash = step(0.5, fract(g.x * 0.5 + 0.25));
        col = mix(col, vec3(0.82, 0.78, 0.42),
                  smoothstep(0.055, 0.015, abs(fract(g.y) - 0.02)) * dash * 0.75);
    } else if (row == 12.0) {
        // The far bank: reeds behind the bays, so the row the frog is working
        // toward is somewhere rather than a dark strip.
        float reed = sin(g.x * 7.0) * 0.5 + 0.5;
        col = mix(vec3(0.07, 0.30, 0.13), vec3(0.11, 0.44, 0.18), reed);
        col = mix(col, palette(iHue + 0.25), 0.10);
        col *= 0.85 + 0.30 * hash21(floor(g * 4.0));
    } else {
        // Safe banks: the start strip and the median.
        col = mix(vec3(0.10, 0.34, 0.14), vec3(0.13, 0.44, 0.18),
                  hash21(floor(g * 5.0)));
        col = mix(col, palette(iHue + 0.3), 0.07);
    }
    // A darker seam between every row so the lanes read as lanes.
    col *= 0.80 + 0.20 * smoothstep(0.0, 0.05, fract(g.y));

    // ---- home bays: notches cut into the top band ----
    if (row == 12.0) {
        for (int i = 0; i < 5; i++) {
            vec4 H = iHome[i];
            vec2 q = g - vec2(H.x, float(12.0) + 0.5);
            float bay = sdBox(q, vec2(0.80, 0.44), 0.18);
            col = mix(col, vec3(0.02, 0.07, 0.16), smoothstep(0.05, 0.0, bay));
            // A lit rim on every bay, so the targets are legible from across
            // the board and an open one pulses with the track.
            col += mix(neon, vec3(0.35, 1.0, 0.55), 0.4)
                 * smoothstep(0.10, 0.0, abs(bay)) * (0.55 + iBass * 0.7);
            if (H.y > 0.5) {
                // A frog already home: seated, facing out, eyes and all.
                float body = sdBox(q, vec2(0.32, 0.24), 0.15);
                float legs = min(length(q - vec2(-0.34, -0.14)),
                                 length(q - vec2(0.34, -0.14))) - 0.12;
                col = mix(col, vec3(0.16, 0.62, 0.20), smoothstep(0.04, 0.0, legs));
                col = mix(col, vec3(0.26, 0.92, 0.32), smoothstep(0.04, 0.0, body));
                float ey = min(length(q - vec2(-0.13, 0.20)),
                               length(q - vec2(0.13, 0.20))) - 0.10;
                col = mix(col, vec3(0.97), smoothstep(0.03, 0.0, ey));
                col = mix(col, vec3(0.04), smoothstep(0.025, 0.0,
                          min(length(q - vec2(-0.13, 0.22)),
                              length(q - vec2(0.13, 0.22))) - 0.045));
                // Arrival flare, fading over about a second.
                col += neon * exp(-max(bay, 0.0) * 6.0) * H.z * 1.6;
            }
        }
    }

    // ---- river traffic: logs and turtles ----
    for (int i = 0; i < 18; i++) {
        vec4 L = iLog[i];
        if (L.z <= 0.0) continue;
        vec2 q = g - vec2(L.x, L.y + 0.5);
        if (abs(q.x) > L.z + 0.6 || abs(q.y) > 0.6) continue;
        if (L.w < 0.5) {
            // Log: end caps plus the grain running along it.
            float body = sdBox(q, vec2(L.z, 0.34), 0.18);
            vec3 bark = mix(vec3(0.36, 0.20, 0.09), vec3(0.50, 0.30, 0.14),
                            0.5 + 0.5 * sin(q.y * 12.0 + q.x * 0.7));
            col = mix(col, bark, smoothstep(0.04, 0.0, body));
            col = mix(col, vec3(0.62, 0.42, 0.22),
                      smoothstep(0.04, 0.0, abs(abs(q.x) - L.z + 0.16) - 0.05));
        } else {
            // Turtles: a row of shells that dip under and come back up.
            float n = floor(L.z + 0.5);
            for (float k = 0.0; k < 3.0; k += 1.0) {
                if (k >= n) break;
                vec2 c = q - vec2((k - (n - 1.0) * 0.5) * 0.94, 0.0);
                // Submerged turtles ride low and go translucent.
                float sub = clamp(L.w - 1.0, 0.0, 1.0);
                float vis = 1.0 - sub * 0.62;
                float dirT = L.y > 0.0 ? 1.0 : 1.0;
                float shell = length(c * vec2(0.94, 1.22)) - 0.46 + sub * 0.14;
                float head = length(c - vec2(0.44 * dirT, 0.0)) - 0.15;
                vec3 sc = mix(vec3(0.24, 0.66, 0.30), vec3(0.10, 0.32, 0.40), sub);
                col = mix(col, sc * 0.78, smoothstep(0.05, 0.0, head) * vis);
                col = mix(col, sc, smoothstep(0.05, 0.0, shell) * vis);
                // Shell plates: a ring and a centre boss, so it reads as a
                // carapace rather than a green dot.
                col = mix(col, sc * 1.28,
                          smoothstep(0.04, 0.0, abs(length(c) - 0.26) - 0.05) * vis);
                col = mix(col, sc * 0.60,
                          smoothstep(0.04, 0.0, length(c) - 0.11) * vis);
            }
        }
    }

    // ---- road traffic ----
    for (int i = 0; i < 20; i++) {
        vec4 C = iCar[i];
        if (C.z <= 0.0) continue;
        vec2 q = g - vec2(C.x, C.y + 0.5);
        if (abs(q.x) > C.z + 0.9 || abs(q.y) > 0.7) continue;
        float dir = C.w < 0.0 ? -1.0 : 1.0;
        float kind = abs(C.w);
        vec3 body = palette(iHue + 0.12 + kind * 0.21);
        if (kind > 2.5) body = vec3(0.85, 0.72, 0.14);      // the tractor
        float chassis = sdBox(q, vec2(C.z, 0.31), 0.10);
        col = mix(col, body, smoothstep(0.04, 0.0, chassis));
        // Cabin set into the leading end, scaled to the vehicle so a short car
        // does not end up as one large windscreen with a rim of paint.
        float cw = min(0.34, C.z * 0.46);
        vec2 cc = vec2(dir * (C.z - cw - 0.07), 0.05);
        col = mix(col, body * 1.22,
                  smoothstep(0.04, 0.0, sdBox(q - cc, vec2(cw, 0.20), 0.06)));
        col = mix(col, vec3(0.05, 0.09, 0.17),
                  smoothstep(0.03, 0.0,
                             sdBox(q - cc - vec2(0.0, 0.03),
                                   vec2(cw * 0.70, 0.10), 0.03)));
        // A darker skirt along the bottom, which is what gives the side view
        // its weight at this size.
        col = mix(col, body * 0.55,
                  smoothstep(0.03, 0.0, sdBox(q - vec2(0.0, -0.22),
                                              vec2(C.z * 0.94, 0.09), 0.04)));
        float wh = min(length(q - vec2(-C.z + 0.32, -0.30)),
                       length(q - vec2(C.z - 0.32, -0.30))) - 0.15;
        col = mix(col, vec3(0.04), smoothstep(0.04, 0.0, wh));
        col += body * smoothstep(0.09, 0.0, abs(chassis)) * (0.25 + iBass * 0.45);
        // Headlights: two tight points with a short throw, not a bloom that
        // swallows the vehicle it belongs to.
        vec2 lp = q - vec2(dir * (C.z + 0.04), -0.04);
        col += vec3(1.0, 0.95, 0.76)
             * exp(-dot(lp * vec2(1.05, 5.0), lp) * 7.0)
             * (0.55 + iTreble * 0.5) * iGlitter;
        col += vec3(1.0, 0.93, 0.70) * 0.30
             * exp(-dot(lp * vec2(0.30, 3.2), lp) * 2.2) * iGlitter;
    }

    // ---- the frog ----
    {
        float hop = iFrog.z;
        vec2 q = g - vec2(iFrog.x, iFrog.y + 0.5 + hop * 0.45);
        if (abs(q.x) < 1.4 && abs(q.y) < 1.4) {
            float dying = iFrog.w;
            // The hop squashes on take-off and stretches at the apex.
            float sq = 1.0 + hop * 0.28;
            vec2 s = q / vec2(1.0 / sq, sq);
            vec3 skin = mix(vec3(0.28, 0.88, 0.30), palette(iHue + 0.45), 0.12);
            if (dying > 0.0) skin = mix(skin, vec3(0.95, 0.25, 0.18), dying);
            float body = sdBox(s, vec2(0.28, 0.30), 0.16);
            // Legs splay out while airborne.
            float spread = 0.30 + hop * 0.26;
            float legs = min(min(length(s - vec2(-spread, -0.22)),
                                 length(s - vec2(spread, -0.22))),
                             min(length(s - vec2(-spread, 0.20)),
                                 length(s - vec2(spread, 0.20)))) - 0.14;
            col = mix(col, skin * 0.80, smoothstep(0.05, 0.0, legs));
            col = mix(col, skin, smoothstep(0.05, 0.0, body));
            // Eyes on top, pupils forward.
            float eyes = min(length(s - vec2(-0.15, 0.28)),
                             length(s - vec2(0.15, 0.28))) - 0.13;
            col = mix(col, vec3(0.96), smoothstep(0.04, 0.0, eyes));
            float pup = min(length(s - vec2(-0.15, 0.31)),
                            length(s - vec2(0.15, 0.31))) - 0.06;
            col = mix(col, vec3(0.04), smoothstep(0.03, 0.0, pup));
            col += skin * smoothstep(0.14, 0.0, abs(body)) * (0.25 + iBeat * 0.7);
            // Splat ring on death.
            if (dying > 0.0) {
                float r = dying * 1.1;
                col += vec3(1.0, 0.5, 0.25)
                     * smoothstep(0.16, 0.0, abs(length(q) - r))
                     * (1.0 - dying);
            }
        }
    }

    col += neon * iBeat * 0.05;
    col = col / (1.0 + col * 0.30);
    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.88)), 1.0);
}
"""


class Frogger(BaseEffect):
    """Frogger, driven by a frog that sweeps the traffic before committing."""

    NAME = 'Frogger'
    AUTHOR = 'unicorn-viz'
    TAGS = ['games', 'arcade', 'neon', 'groovy', 'energetic']
    PING_PONG_FRIENDS = [
        'Super Mario', 'Mario Kart', 'Donkey Kong', 'Neon Pac', 'Breakout',
        'Q*bert',
    ]

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.2)),
            'glitter': float(self.config.get('glitter', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._glitter = max(0.0, float(self.parameters['glitter']))

        # --- road: one direction and pace per lane, vehicles spaced round it --
        self._cars: list[dict[str, float]] = []
        for row in range(_ROAD_LO, _ROAD_HI + 1):
            sgn = 1.0 if (row % 2 == 0) else -1.0
            v = sgn * float(self.rng.uniform(2.6, 5.2))
            kind = float(self.rng.integers(1, 4))
            half = (0.55, 0.55, 1.05, 1.35)[int(kind)]
            # Size the lane so a window is guaranteed to exist. A vehicle
            # blocks `v * _LOOK + 2 * (half + margin)` tiles of the frog's
            # decision, and if that exceeds the spacing the lane can never
            # open -- four long vehicles at speed did exactly that, and the
            # frog sat on row 2 for a full 90 s run. The 1.3 leaves a window
            # with some width to it rather than a hairline.
            blocked = abs(v) * _LOOK + 2.0 * (half + 0.55)
            n = int(np.clip((2.0 * _HALF_W) / (blocked * 1.3), 2, 4))
            gap = (2.0 * _HALF_W) / n
            x0 = float(self.rng.uniform(0.0, gap))
            for k in range(n):
                self._cars.append({'x': -_HALF_W + x0 + k * gap, 'row': float(row),
                                   'half': half, 'v': v, 'kind': kind})
        # --- river: logs and turtle rafts ------------------------------------
        self._logs: list[dict[str, float]] = []
        for row in range(_RIVER_LO, _RIVER_HI + 1):
            sgn = 1.0 if (row % 2 == 1) else -1.0
            v = sgn * float(self.rng.uniform(1.7, 3.4))
            turtles = bool(self.rng.random() < 0.4)
            # Logs long enough that a lane is more raft than water; below about
            # half the lane covered the frog spends most of a crossing waiting.
            half = 3.0 if turtles else float(self.rng.uniform(1.8, 3.2))
            n = int(self.rng.integers(3, 5))
            gap = (2.0 * _HALF_W) / n
            x0 = float(self.rng.uniform(0.0, gap))
            for k in range(n):
                self._logs.append({
                    'x': -_HALF_W + x0 + k * gap, 'row': float(row),
                    'half': half, 'v': v, 'turtle': float(turtles),
                    # Turtle rafts dive on their own schedule, so a lane is
                    # never uniformly safe or uniformly lethal.
                    'dive': float(self.rng.uniform(0.0, 1.0)),
                    'drate': float(self.rng.uniform(0.055, 0.10)),
                })

        self._home_x = np.linspace(-8.0, 8.0, _NHOME)
        self._home = np.zeros(_NHOME, dtype=np.float64)
        self._home_glow = np.zeros(_NHOME, dtype=np.float64)
        # Start with a bay or two already taken so a fresh activation does not
        # always look like the very first frog of the game.
        for i in range(_NHOME):
            if self.rng.random() < 0.28:
                self._home[i] = 1.0

        self._fx = float(self.rng.integers(-4, 5))
        self._fy = 0.0
        self._hop_from = (self._fx, self._fy)
        self._hop_to = (self._fx, self._fy)
        self._hop_t = 1.0           # 1.0 = landed
        self._wait = float(self.rng.uniform(0.1, 0.5))
        self._dying = 0.0
        self._ride = 0.0            # world speed the frog inherits from a log
        self._pace = 1.0            # traffic speed multiplier, set each frame

        self._tbl_car = np.zeros((_NCAR, 4), dtype='f4')
        self._tbl_log = np.zeros((_NLOG, 4), dtype='f4')
        self._tbl_home = np.zeros((_NHOME, 4), dtype='f4')

    # ------------------------------------------------------------- the sweep
    def _car_free(self, x: float, row: int, when: float,
                  margin: float = 0.55) -> bool:
        """Will the tile at *x* on *row* still be clear *when* seconds from now?

        Sweeps each car's span over the interval rather than sampling its
        position at the landing instant: a car 5 tiles long at 5 tiles/s covers
        its own length in a single hop, so a point sample misses it outright.
        """
        for c in self._cars:
            if int(c['row']) != row:
                continue
            a = c['x']
            b = c['x'] + c['v'] * self._pace * when
            lo, hi = (a, b) if a <= b else (b, a)
            if lo - c['half'] - margin < x < hi + c['half'] + margin:
                return False
        return True

    def _float_at(self, x: float, row: int, when: float = 0.0
                  ) -> dict[str, float] | None:
        """The floater covering *x* on *row* after *when* seconds, if any."""
        for f in self._logs:
            if int(f['row']) != row:
                continue
            fx = f['x'] + f['v'] * self._pace * when
            if f['turtle'] > 0.5:
                # A raft that has gone under carries nothing, and a raft that
                # will have gone under by the time the frog arrives is worse.
                dive = (f['dive'] + f['drate'] * self._pace * when) % 1.0
                if math.sin(dive * math.tau) > 0.62:
                    continue
                span = f['half'] * 0.46
            else:
                span = f['half']
            if abs(x - fx) < span + 0.42:
                return f
        return None

    def _can_stand(self, x: float, row: int, hop_t: float,
                   drift: float = 0.0) -> bool:
        """Can the frog be at *x* on *row*, arriving *hop_t* seconds from now?

        The two halves of the board ask opposite questions. On the road the
        frog stands still while the traffic moves, so the tile has to stay
        clear for the hop *and* the dwell that follows it. On the river the
        frog rides whatever it lands on, so the dwell takes care of itself and
        all that matters is that something is there at the moment of landing.

        Sweeping the river with the road's horizon looks for a log 0.8 s
        downstream of where the frog will actually arrive -- two tiles off at
        river speeds. That one mistake accounted for nine of the eleven river
        deaths in a four-run sample, all of them on the first row of water.

        *drift* is the frog's own speed, non-zero only when it is already
        riding, so that a raft it is standing on is not mistaken for one
        floating away from it.
        """
        if _RIVER_LO <= row <= _RIVER_HI:
            here = x + drift * self._pace * hop_t
            return self._float_at(here, row, hop_t) is not None
        if _ROAD_LO <= row <= _ROAD_HI:
            return self._car_free(x, row, hop_t + _DWELL)
        return abs(x) < _HALF_W - 1.5

    def _choose(self) -> None:
        """Pick this hop: up if the landing is safe, otherwise sidestep or hold."""
        row = int(round(self._fy))
        nxt = row + 1
        # Aim at the nearest bay that is still open once the river is behind.
        if nxt >= _HOME_ROW:
            open_bays = [i for i in range(_NHOME) if self._home[i] < 0.5]
            if not open_bays:
                self._home[:] = 0.0
                open_bays = list(range(_NHOME))
            # Aim at the bay the current drifts toward, not the one that is
            # nearest right now: the log keeps moving while the frog decides,
            # and chasing the bay behind it is how it gets carried off the
            # edge. That accounted for six of eighteen deaths in a 4x120 s
            # sample, all of them on the last row of water.
            soon = self._fx + self._ride * self._pace * 1.0
            tgt = min(open_bays, key=lambda i: abs(self._home_x[i] - soon))
            dx = self._home_x[tgt] - self._fx
            # Running out of board beats any bay preference.
            if abs(self._fx) > _HALF_W - 4.0:
                dx = -self._fx
            # Wide enough that a drifting log cannot skip past alignment
            # between two decision instants -- it covers over a tile of ground
            # in that time, so a 0.6 window is regularly stepped straight over.
            if abs(dx) > 1.6:
                # Only shuffle sideways if there is still something to land on;
                # otherwise ride and wait, because the log drifts into line.
                side = self._fx + math.copysign(1.0, dx)
                if self._can_stand(side, row, _HOP_T, drift=self._ride):
                    self._start_hop(side, self._fy)
                    return
                # Blocked on the last row of water. Returning here just leaves
                # the frog riding toward the edge -- dropping back to the lane
                # below buys another pass at the bays on a log going the other
                # way, and row 11 was both the longest stay on the board and a
                # third of all deaths before this fell through.
                if self._can_stand(self._fx, row - 1, _HOP_T):
                    self._start_hop(self._fx, self._fy - 1.0)
                return
            self._start_hop(float(self._home_x[tgt]), float(_HOME_ROW))
            return

        if self._can_stand(self._fx, nxt, _HOP_T):
            self._start_hop(self._fx, self._fy + 1.0)
            return

        # Blocked. Sidestep toward a neighbouring column that is clear, and if
        # neither is, simply hold -- waiting is a legitimate move here.
        for dx in (1.0, -1.0) if self.rng.random() < 0.5 else (-1.0, 1.0):
            nx = self._fx + dx
            if abs(nx) > _HALF_W - 2.0:
                continue
            # The sidestep itself has to be survivable, and it only helps if
            # it opens the row above.
            if (self._can_stand(nx, row, _HOP_T, drift=self._ride)
                    and self._can_stand(nx, nxt, _HOP_T)):
                self._start_hop(nx, self._fy)
                return
        # Nowhere forward and nowhere sideways: drop back a row if that is
        # survivable. Retreating is how a player survives a wall of traffic.
        back = self._fy - 1.0
        if back >= 0.0 and self._can_stand(self._fx, int(back), _HOP_T):
            self._start_hop(self._fx, back)
            return
        self._wait = 0.08

    def _start_hop(self, x: float, y: float) -> None:
        self._hop_from = (self._fx, self._fy)
        self._hop_to = (x, y)
        self._hop_t = 0.0

    def _die(self) -> None:
        self._dying = 1e-3
        self._hop_t = 1.0

    def _respawn(self) -> None:
        self._fx = float(self.rng.integers(-4, 5))
        self._fy = 0.0
        self._hop_from = self._hop_to = (self._fx, self._fy)
        self._hop_t = 1.0
        self._dying = 0.0
        self._ride = 0.0
        self._wait = 0.25

    # ------------------------------------------------------------------ sim
    def update(self, dt: float, audio: AudioData) -> None:
        """Advance the traffic, then let the frog decide what to do about it."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 1.8))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 1.8))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 1.8))
        beat_now = audio.beat > 0.5
        if beat_now:
            self._beat = 1.0
        self._beat = max(0.0, self._beat - dt * 3.6)

        self._pace = (max(0.05, float(self.parameters['speed']))
                      * (0.72 + self._bass * 0.55))
        pace = self._pace
        wrap = _HALF_W + 4.0
        for c in self._cars:
            c['x'] += c['v'] * pace * dt
            if c['x'] > wrap:
                c['x'] = -wrap
            elif c['x'] < -wrap:
                c['x'] = wrap
        for f in self._logs:
            f['x'] += f['v'] * pace * dt
            if f['x'] > wrap:
                f['x'] = -wrap
            elif f['x'] < -wrap:
                f['x'] = wrap
            if f['turtle'] > 0.5:
                f['dive'] = (f['dive'] + f['drate'] * dt) % 1.0
        self._home_glow *= math.exp(-dt * 1.4)

        if self._dying > 0.0:
            self._dying = min(1.0, self._dying + dt * 1.1)
            if self._dying >= 1.0:
                self._respawn()
            return

        if self._hop_t < 1.0:
            self._hop_t = min(1.0, self._hop_t + dt / _HOP_T)
            u = self._hop_t
            self._fx = self._hop_from[0] + (self._hop_to[0] - self._hop_from[0]) * u
            self._fy = self._hop_from[1] + (self._hop_to[1] - self._hop_from[1]) * u
            if self._hop_t >= 1.0:
                self._land()
            return

        # On the ground: ride whatever is underneath, then wait for the beat.
        row = int(round(self._fy))
        if _RIVER_LO <= row <= _RIVER_HI:
            self._fx += self._ride * pace * dt
            self._hop_from = self._hop_to = (self._fx, self._fy)
            if abs(self._fx) > _HALF_W - 1.0:
                self._die()          # carried off the side of the screen
                return
            if self._float_at(self._fx, row) is None:
                self._die()          # the raft went under
                return
        elif (_ROAD_LO <= row <= _ROAD_HI
              and not self._car_free(self._fx, row, 0.0, margin=0.28)):
            self._die()
            return

        # If this tile is about to stop being worth standing on, move now
        # rather than holding out for the next beat.
        panic = not self._can_stand(self._fx, row, _PANIC, drift=self._ride)
        self._wait -= dt
        # Hop on the beat when one is close, but never let the beat gate hold
        # the frog still for longer than it takes to miss a window.
        if self._wait <= 0.0 and (beat_now or panic or self._wait < -0.16):
            self._choose()
            self._wait = 0.08 + 0.15 / (0.5 + self._bass)

    def _land(self) -> None:
        """Resolve what the frog landed on."""
        self._fx, self._fy = self._hop_to
        row = int(round(self._fy))
        if row == _HOME_ROW:
            i = int(np.argmin(np.abs(self._home_x - self._fx)))
            self._home[i] = 1.0
            self._home_glow[i] = 1.0
            self._respawn()
            return
        if _RIVER_LO <= row <= _RIVER_HI:
            f = self._float_at(self._fx, row)
            if f is None:
                self._die()
                return
            self._ride = f['v']
            return
        self._ride = 0.0
        if (_ROAD_LO <= row <= _ROAD_HI
                and not self._car_free(self._fx, row, 0.0, margin=0.28)):
            self._die()

    def render(self) -> None:
        """Upload the frame's spans; the shader only rasterizes them."""
        self._tbl_car[:] = 0.0
        for i, c in enumerate(self._cars[:_NCAR]):
            self._tbl_car[i] = (c['x'], c['row'], c['half'],
                                math.copysign(c['kind'], c['v']))
        self._tbl_log[:] = 0.0
        for i, f in enumerate(self._logs[:_NLOG]):
            if f['turtle'] > 0.5:
                # w carries 1 + how far under this raft has gone.
                sub = max(0.0, math.sin(f['dive'] * math.tau)) ** 2
                self._tbl_log[i] = (f['x'], f['row'], f['half'], 1.0 + sub)
            else:
                self._tbl_log[i] = (f['x'], f['row'], f['half'], 0.0)
        for i in range(_NHOME):
            self._tbl_home[i] = (self._home_x[i], self._home[i],
                                 self._home_glow[i], 0.0)

        # A sine arc, so the frog leaves and meets the ground at a real angle
        # instead of sliding flat between tiles.
        arc = math.sin(self._hop_t * math.pi) if self._hop_t < 1.0 else 0.0

        p = self._prog
        p['iResolution'].value = (float(self.width), float(self.height))
        p['iTime'].value = float(self.time)
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iHue'].value = self._hue
        p['iGlitter'].value = self._glitter
        p['iCar'].write(self._tbl_car.tobytes())
        p['iLog'].write(self._tbl_log.tobytes())
        p['iHome'].write(self._tbl_home.tobytes())
        p['iFrog'].value = (float(self._fx), float(self._fy), float(arc),
                            float(self._dying))
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()

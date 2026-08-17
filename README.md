# games-01 — Unicorn Viz effect pack

**Version 0.13.2**


arcade / game-like visuals: Breakout, Neon Pac, Galaga, Joust, Tetris,
Missile Command, Donkey Kong, Q*bert.

Each effect subclasses `unicornviz.effects.base.BaseEffect` and is auto-discovered.

## Changelog

- **0.13.2** — killed the "ASCII dots": the regular grid of tiny marks visible
  across Donkey Kong, Galaga, Joust, Missile Command, Neon Pac, Q\*bert and
  Tetris. Every one used the sparkle idiom
  `hash21(floor(frag * 0.5) + floor(t * 20.0))`, which feeds a term reaching
  ~200000 into a hash whose first multiply is x123.34 — past float32's usable
  mantissa, so the "random" field collapses into a fixed lattice. Measured
  over a 40x40 patch: **1197 distinct values at t=0, 15 at t=600s, and 1 at
  t=10000** — and since `self.time` starts at a random value up to 10000,
  essentially every instance ran with a completely dead sparkle. `hash21` now
  wraps its input first (`mod(p, 512.0)`), restoring 1234/1600 distinct values
  at t=10000. One line per file; no visual design changed.

- **0.13.1** — Q\*bert: characters now render at full opacity. Every sprite was
  composited with `col += drawX(...)`, i.e. *added* to whatever was already
  there, so the cube underneath showed through and washed the cast out. Each
  character routine now tracks a coverage mask alongside its colour and returns
  `vec4(rgb, coverage)`; `main()` composites with `mix()` so a character fully
  replaces the pixel it covers. Highlights that sit inside an already-covered
  body stay additive (they brighten without widening the silhouette), while
  parts that extend past the body — Coily's tongue, Slick's hair tuft, the
  disc's rim lights — carry their own coverage. The hero's death blink now
  drops coverage too, so he flickers out instead of turning into a dark
  silhouette.

- **0.13.0** — Q\*bert 2.0. The whole cast was two ellipses: the hero was a
  plain orange blob with two dots for eyes and the single enemy was the same
  blob in purple — no snout, no Coily, none of the arcade cast.

  **Characters**, each a real actor in the simulation rather than set dressing:
  - **Q\*bert** — orange, with the signature tubular snout that swings to
    whichever side he is hopping toward, big eyes whose pupils track with it,
    feet, and a body that stretches through the hop arc.
  - **Coily** — arrives as an egg at the apex, bounces blindly downward, then
    hatches into the purple snake: a coiled spring body whose coils spread
    apart mid-hop and compress on landing, a head that rides the stretch, and
    a forked tongue that flicks. Once hatched he hunts Q\*bert directly.
  - **Red balls** — bounce down the pyramid on their own and kill on contact.
  - **Slick & Sam** — little green cube-thieves with waggling hair tufts that
    hop downward *un-colouring* finished cubes (241 cubes stolen back across a
    300s sim), so a course can be actively undone while it is being solved.
  - **Flying discs** — one either side. With Coily adjacent, Q\*bert leaps to a
    disc, Coily dives off after him, and the disc is spent (20 escapes/300s).

  **Six courses**, cycled without immediate repeats, in three rulesets:
  `once` (one hop per cube), `two` (two visits through an intermediate
  colour), and `toggle` (landing flips a cube back off, so the AI has to avoid
  undoing its own work). Pyramids run 5–8 rows, each with its own cube palette
  so a switch is unmistakable. The lattice is sized per row count and the
  occupied band — including the character standing on the apex — is centred,
  verified to fit at both 16:9 and 4:3 for every course. Cube and character
  scale now follow the course size instead of being hard-coded.

  **Beat pulse** — every beat sends a wave down the pyramid, lifting and
  flaring cubes in sequence from the apex outward. The lift is computed once
  on the CPU and baked into the uploaded cube position, then applied to
  whatever is standing on that row, so characters ride the wave with their
  platform instead of detaching from it.

  **Drip** — cubes redrawn as proper isometric solids with a glossy sweep
  across the top face, a neon rim on the diamond outline, and neon paint
  running down the side faces (treble-reactive).

  Caught three rendering bugs during the build, all the same class — an
  unmasked additive term painting a sprite's rectangular bounding box over
  the scene: Q\*bert's ambient glow, and the pulse-wave glow in the cube loop
  (which banded whole rows). Also caught `iCoilyHop` declared and assigned but
  never referenced in the shader — the exact crash-on-load pattern that hit
  Wingsuit Dive and Sun Ship 3000 — and wired it into Coily's hop stretch.
  Verified with per-course renders, a high-iTime stress test, and a 300s sim
  exercising all six courses, all three rulesets, 21 hatches, 14 deaths, and
  20 disc escapes.

- **0.12.0** — Donkey Kong 2.0, Tetris 2.0, and a Missile Command bug sweep.

  **Donkey Kong 2.0** — the cast was the problem: the hero stood 0.49 world
  units tall (half the screen height, spanning three girder gaps), and Kong
  was drawn with his feet on the top girder at y≈0.43 so his head sat at
  y≈0.73 when the visible range ends at 0.5 — he was rendered every frame,
  entirely off-screen, which is why there appeared to be no Kong at all.
  - Hero rebuilt at ~0.10 units as a real sprite: cap with a brim, moustache,
    overalls with straps, boots, swinging arms and a gait that changes
    between walking and climbing.
  - Kong rebuilt at ~0.175 units and repositioned to fit: fur body, pale
    chest patch that swells on the beat, muzzle, nostrils, brow, tracking
    eyes, an idle arm that pounds his chest while raging and a throwing arm
    that winds up and over for every barrel.
  - Added a damsel on the far end of the top girder (waving for help), a
    burning oil drum where barrels retire, and victory hearts that rise
    between the pair when the hero tops out.
  - Added a hammer power-up: it appears on a girder, the hero detours to
    grab it, and while it is up he hunts barrels and smashes them into spark
    bursts instead of dying. Classic rule kept — no climbing while carrying
    it, so taking it is a genuine trade-off the AI makes.
  - Barrels now sometimes take a ladder down instead of always rolling to
    the end, and a quarter of them are flaming variants.
  - Fixed the rig only spanning x∈[-0.55,0.55] while a 16:9 screen spans
    ±0.89 — the same width bug found in Missile Command. The simulation now
    keeps x as a screen fraction and converts at upload, so the girders
    reach both edges at any aspect ratio.
  - Fixed the ladders rendering as solid slabs: the rung mask subtracted a
    constant inside `smoothstep`, lighting ~80% of each cell instead of
    drawing a thin line.

  **Tetris 2.0** — it was slow because gravity ran at 1.6 cells/sec, so a
  piece took ~12s to cross the well. Rebuilt around fast alignment plus a
  hard drop: the AI snaps to its planned rotation/column at 26 steps/sec,
  pauses briefly, then slams the piece down. Measured 3.4 pieces/sec (was
  well under one), 74 lines in 60 seconds.
  - Every drop leaves streak trails down the columns it fell through and
    lands with a shockwave ring and a screen-shake kick.
  - Line clears fire a horizontal disintegration beam across each full row
    before it collapses.
  - Added a level counter driving gravity, shown as a lit segment ladder
    beside the well, and top-out detection — a jammed board now detonates
    and restarts instead of silently stacking forever.
  - Futuristic presentation: receding perspective grid and falling data-rain
    behind the well, corner brackets and a travelling scan line on the
    frame, next-piece hologram panel, and blocks redrawn as bevelled shells
    around a pulsing energy core with a travelling scanline.

  **Missile Command** — three real bugs:
  - Batteries fired at missiles that were not on screen yet. Wave members
    spawn with a negative `prog` so they stagger in, but `_current_pos`
    clamps prog to 0, so an unlaunched missile still reported a position at
    its spawn altitude and the targeting scan happily led it. Targeting now
    runs off a visible-only list; verified 0 such shots across a 180s sim
    (previously routine).
  - Buildings did not visibly die when hit — a destroyed city kept its full
    silhouette and merely changed colour. Cities now actually come down:
    each building collapses to a jagged chewed-off stub, the ground craters
    and scorches, embers wink in the wreckage and smoke drifts up. The
    background skyline was also drawing *taller* than the defended cities
    (0.14 vs 0.052), hiding the wreckage behind it; it is now clearly
    shorter so a levelled city reads instantly.
  - Cityscape scenes changed too rarely — the timer is now 10–18s (was
    24–40s), giving 13 scene changes per 180s instead of ~5.
  - Added UFOs: saucers with domes and chasing rim lights drift in from
    either edge, bomb the ground line as they pass, are targeted by the
    batteries (led along their drift), and burst when caught in a blast.

  All three verified with real offscreen renders across quiet/moderate/
  loud-beat audio, a high-iTime stress test, and headless simulations
  confirming actual gameplay rather than just "doesn't crash" — Donkey Kong:
  31 climbs completed, 11 hammer grabs, 13 barrels taking ladders; Tetris:
  204 hard drops, level 10 reached, no top-outs; Missile Command: 12 cities
  destroyed, 20 UFOs, 27 bombs dropped, 0 off-screen shots.

- **0.11.1** — Missile Command: a plain uniform draw of the 4 cityscape
  styles on activation had a 1-in-4 chance of repeating whatever style the
  previous activation last showed, which read as "it didn't change" even
  though the pick was genuinely random. Added a class-level `_last_style`
  tracker (updated on every activation and every in-session style switch) so
  the startup draw always excludes the immediately-previous style. Verified
  across 20 consecutive fresh instantiations: zero back-to-back repeats,
  still an even spread across all 4 styles.

- **0.11.0** — Missile Command cityscape overhaul:
  - Fixed a coordinate-space bug where the cityscape/backdrop only spanned a
    fraction of the screen — the shader's world space was non-centred
    (`p = vec2(uv.x * A, uv.y)`, range `[0, A]`) while city/battery x
    positions were plain `[0, 1]` fractions, so on any window wider than
    square the whole skyline sat bunched near the left edge. World space is
    now centred (`p = (uv - 0.5) * vec2(A, 1.0)`) and every simulation
    position is converted to it on upload; verified via offscreen renders
    that skyline pixel coverage now reaches both screen edges.
  - Cities went from a flat 5-building silhouette to a 7-building skyline
    with varied width/height, a lit window grid, and a continuous two-layer
    procedural background skyline that fills the ground across the entire
    width (not just at the six named city markers), so the horizon is never
    sparse.
  - Added four cycling cityscape scene styles — classic neon (the original
    look, width-fixed), chrome spires (glinting rooftop antennae + drip
    streaks), circuit ruins (glowing horizontal trace lines), and drip fade
    (heavier, faster-flowing neon paint streaks) — switched whenever the
    city/battery line is fully wiped and rebuilt, and periodically on a
    randomised 24-40s timer, so a long session keeps evolving. Gameplay,
    missiles, batteries, and blasts are unchanged.
  - Verified via offscreen renders per style/scenario (quiet/moderate/loud
    beat), a high-iTime stress test, and a 180s headless sim confirming
    style transitions actually occur during real play.

- **0.10.0** — four new effects, each a real CPU simulation rendered as neon
  (same architecture as the rest of the pack — no procedural-only fakes):
  - **Tetris** — a real 10x20 well with the seven tetromino shapes, an AI
    that evaluates every rotation/column (minimising holes/height/bumpiness,
    rewarding line clears) and executes the best placement, line-clear
    shrapnel bursts, and a glowing next-piece preview.
  - **Missile Command** — real incoming-missile trajectories, three
    auto-firing defense batteries with simple lead-targeting, expanding
    interceptor blasts that chain-destroy threats, cities/batteries that
    crater on impact and quietly rebuild once the whole line is wiped.
  - **Donkey Kong** — five zig-zag girder platforms with ladders, a hero who
    paths toward the next ladder up and hops over rolling barrels, real
    barrel physics rolling downhill and dropping a level at each girder's
    low end, Kong throwing barrels from the top on a timer.
  - **Q\*bert** — a real 7-row/28-cube isometric pyramid, diagonal hopping
    toward the nearest not-yet-recoloured cube, a chasing enemy that spawns
    partway through each level, cube-colour level-clear resets.

  All four verified via real offscreen renders (moderngl standalone context)
  across quiet/moderate/loud-beat audio scenarios, headless simulation runs
  confirming actual gameplay (line clears, interceptions, barrel deaths,
  level clears — not just "doesn't crash"), and an explicit high-iTime
  stress test per the lesson learned building flying-01. Caught and fixed a
  coordinate-space bug in Donkey Kong (girder endpoints were defined in
  [0,1] UV space but the shader's world space is centred at zero, pushing
  the whole rig off-screen) and two stripped/mismatched-array-size uniform
  bugs (Tetris's shard array size didn't match its GLSL declaration; both
  Tetris and Missile Command had uniforms set from Python but never
  referenced in the shader).
- **0.9.1** — Joust: fixed birds piling up and camping near the screen's top
  edge. Two causes: the "get above the target" altitude AI had no ceiling
  awareness (both sides could spiral upward chasing each other), and the
  joust level-clash bounce added unclamped vertical velocity that could stack
  past the normal flap speed cap across repeated clashes, rocketing a bird
  into the ceiling on momentum alone. Added a soft-ceiling target clamp, a
  velocity-aware flap gate (projects the coast-to-apex height before allowing
  another flap), general pairwise separation between all birds (previously
  only player-vs-enemy resolved), and a proper clamp on the clash bounce.
- **0.9.0** — initial versioned release.

# games-01 — Unicorn Viz effect pack

**Version 0.11.1**


arcade / game-like visuals: Breakout, Neon Pac, Galaga, Joust, Tetris,
Missile Command, Donkey Kong, Q*bert.

Each effect subclasses `unicornviz.effects.base.BaseEffect` and is auto-discovered.

## Changelog

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

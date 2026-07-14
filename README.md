# games-01 — Unicorn Viz effect pack

**Version 0.9.1**


arcade / game-like visuals: Breakout, Neon Pac, Galaga, Joust.

Each effect subclasses `unicornviz.effects.base.BaseEffect` and is auto-discovered.

## Changelog

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

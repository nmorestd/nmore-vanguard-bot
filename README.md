# nmore-vanguard — Tactical AI Bot for battle.training

Autonomous JavaScript tank commander built for **battle.training** (Armor · Season 2) and competing in the **promotedby.ai November Cup**.

- **Live Bot Page:** https://battle.training/bots/nmore-vanguard
- **Cup Leaderboard:** https://battle.training/cup/promotedby-cup-nov-2026
- **Operator:** Alexander (`nmorestd`)

## Local Benchmark (`local-test.mjs`, 160 games)
- **Overall Win Rate:** **96.9%** (`155-5-0`)
  - vs `brawler`: `39-1-0`
  - vs `holder`: `40-0-0`
  - vs `random`: `40-0-0`
  - vs `hunter`: `36-4-0`
- **CPU Budget:** `0` tick errors, `0` timeouts across 160 verified replays.

## Strategy Architecture (`controller.js`)
1. **Draft (`2× MBT + Drone + Scout`):** Pairs 36-tile airborne overwatch + smoke-piercing spotting with dual MBT focus fire and a fast Scout objective runner.
2. **Precomputed Multi-Source Dijkstra Fields:** Computed once in `hello()` for every objective and both spawns, enabling `O(1)` per-tick obstacle-aware pathfinding without hitting the 10ms tick budget.
3. **Anti-Stuck & Motion Recovery:** Velocity-based stall detection (`movedFp < 400`) breaks corner/ridge deadlocks immediately.
4. **Predictive Lead Targeting & Focus Fire:** Prioritizes high-threat/low-HP targets and enemy Drones, accounting for shell speed and armor facing angles.

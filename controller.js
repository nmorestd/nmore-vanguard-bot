// battle.training controller — Iteration 3 (Apex Vanguard)
// Deterministic, zero-timeout ES2023 controller for Season 2 ("Drafted") & Urban preview.
// Guaranteed O(1) per-tick navigation via precomputed Dijkstra fields + lazy round-robin dynamic field.

const FP = 65536;
const FP_SHIFT = 16;
const FULL = 4096;
const HALF = 2048;

const COS = new Float64Array(FULL);
const SIN = new Float64Array(FULL);
for (let i = 0; i < FULL; i++) {
  const rad = (i * 2 * Math.PI) / FULL;
  COS[i] = Math.cos(rad);
  SIN[i] = Math.sin(rad);
}

const DIRS8 = [
  [1, 0, 10], [-1, 0, 10], [0, 1, 10], [0, -1, 10],
  [1, 1, 14], [1, -1, 14], [-1, 1, 14], [-1, -1, 14],
];

const MAX_WH = 16384;
const sharedHeapIdx = new Int32Array(MAX_WH + 8);
const sharedHeapKey = new Int32Array(MAX_WH + 8);
const bfsQueue = new Int32Array(MAX_WH + 8);
const bfsVisited = new Int32Array(MAX_WH);
let bfsStamp = 1;

let rules = null;
let map = null;
let side = 0;
let W = 128;
let H = 128;
let WH = 16384;
let passableGrid = null;
let tileMoveCost = null;
let objFields = [];
let homeSpawn = [0, 0];
let enemySpawn = [0, 0];
let homeField = null;
let enemyField = null;

// Per-match runtime state (reset in hello)
let mem = new Map();
let unitState = new Map();
let lastSayTick = -100;
let prevMyScore = 0;
let prevEnemyScore = 0;
let smoothEnemyRate = 0;
let smoothMyRate = 0;
let lastBfsTick = -99;

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function angNorm(a) {
  return ((Math.round(a) % FULL) + FULL) & (FULL - 1);
}

function angDiff(a, b) {
  const d = (b - a) & (FULL - 1);
  return d >= HALF ? d - FULL : d;
}

function angleTo(ax, ay, bx, by) {
  return angNorm((Math.atan2(by - ay, bx - ax) * FULL) / (2 * Math.PI));
}

function distFp(ax, ay, bx, by) {
  return Math.hypot(bx - ax, by - ay);
}

function centreFp(tx) {
  return (tx << FP_SHIFT) + (FP >> 1);
}

function isTilePassable(r, t) {
  const spd = r.terrainSpeed ? r.terrainSpeed[t] : undefined;
  if (spd !== undefined) return spd > 0;
  return t !== 2 && t !== 4 && t !== 6 && t !== 8;
}

function blocksDirect(t) {
  return t === 4 || t === 6 || t === 8;
}

function heightBilinear(m, x, y) {
  const fx = clamp(x, 0, (m.width << FP_SHIFT) - 1);
  const fy = clamp(y, 0, (m.height << FP_SHIFT) - 1);
  const tx = fx >> FP_SHIFT;
  const ty = fy >> FP_SHIFT;
  const tx1 = tx + 1 < m.width ? tx + 1 : tx;
  const ty1 = ty + 1 < m.height ? ty + 1 : ty;
  const rx = (fx & 0xffff) / FP;
  const ry = (fy & 0xffff) / FP;
  const h00 = m.heights[ty * m.width + tx];
  const h10 = m.heights[ty * m.width + tx1];
  const h01 = m.heights[ty1 * m.width + tx];
  const h11 = m.heights[ty1 * m.width + tx1];
  return (h00 * (1 - rx) + h10 * rx) * (1 - ry) + (h01 * (1 - rx) + h11 * rx) * ry;
}

function groundAt(m, x, y) {
  if (m.ground) {
    const tx = clamp(x >> FP_SHIFT, 0, m.width - 1);
    const ty = clamp(y >> FP_SHIFT, 0, m.height - 1);
    return m.ground[ty * m.width + tx];
  }
  return heightBilinear(m, x, y);
}

function findNearestPassable(tx, ty) {
  tx = clamp(tx, 0, W - 1);
  ty = clamp(ty, 0, H - 1);
  if (passableGrid[ty * W + tx]) return [tx, ty];
  for (let r = 1; r < 45; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (Math.abs(dx) !== r && Math.abs(dy) !== r) continue;
        const nx = tx + dx;
        const ny = ty + dy;
        if (nx >= 0 && ny >= 0 && nx < W && ny < H && passableGrid[ny * W + nx]) {
          return [nx, ny];
        }
      }
    }
  }
  return [tx, ty];
}

/** Precomputed full-map Dijkstra distance field (runs only in hello!) */
function buildDistanceField(gx, gy) {
  const dist = new Int32Array(WH);
  dist.fill(1e9);
  const startGoal = findNearestPassable(gx, gy);
  const startIdx = startGoal[1] * W + startGoal[0];
  dist[startIdx] = 0;

  let heapSize = 0;
  const push = (idx, key) => {
    let i = ++heapSize;
    while (i > 1) {
      const p = i >> 1;
      if (sharedHeapKey[p] <= key) break;
      sharedHeapIdx[i] = sharedHeapIdx[p];
      sharedHeapKey[i] = sharedHeapKey[p];
      i = p;
    }
    sharedHeapIdx[i] = idx;
    sharedHeapKey[i] = key;
  };

  const pop = () => {
    const top = sharedHeapIdx[1];
    const lastIdx = sharedHeapIdx[heapSize];
    const lastKey = sharedHeapKey[heapSize--];
    let i = 1;
    while ((i << 1) <= heapSize) {
      let c = i << 1;
      if (c + 1 <= heapSize && sharedHeapKey[c + 1] < sharedHeapKey[c]) c++;
      if (lastKey <= sharedHeapKey[c]) break;
      sharedHeapIdx[i] = sharedHeapIdx[c];
      sharedHeapKey[i] = sharedHeapKey[c];
      i = c;
    }
    if (heapSize > 0) {
      sharedHeapIdx[i] = lastIdx;
      sharedHeapKey[i] = lastKey;
    }
    return top;
  };

  push(startIdx, 0);

  while (heapSize > 0) {
    const curCost = sharedHeapKey[1];
    const u = pop();
    if (curCost > dist[u]) continue;
    const ux = u % W;
    const uy = (u - ux) / W;

    for (let d = 0; d < 8; d++) {
      const dx = DIRS8[d][0];
      const dy = DIRS8[d][1];
      const nx = ux + dx;
      const ny = uy + dy;
      if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
      const v = ny * W + nx;
      if (!passableGrid[v]) continue;
      if (dx !== 0 && dy !== 0) {
        if (!passableGrid[uy * W + nx] || !passableGrid[ny * W + ux]) continue;
      }
      const stepCost = (DIRS8[d][2] * tileMoveCost[v]) >> 4;
      const dh = Math.abs(map.heights[u] - map.heights[v]);
      const slopePenalty = dh > 4 ? (dh >> 1) : 0;
      const nc = curCost + stepCost + slopePenalty;
      if (nc < dist[v]) {
        dist[v] = nc;
        push(v, nc);
      }
    }
  }
  return dist;
}

/** Check if a direct shell from shooter u to (tx, ty) has clear line of fire */
function hasClearShot(u, spec, tx, ty) {
  const dist = distFp(u.x, u.y, tx, ty);
  let maxR = spec.gun.maxRange;
  const ha = rules.heightAdvantage;
  const z0 = groundAt(map, u.x, u.y) + spec.eyeHeight;
  const ztGround = groundAt(map, tx, ty);
  if (ha && (groundAt(map, u.x, u.y) - ztGround) >= ha.minDelta) {
    maxR = Math.round((maxR * ha.rangeBonus) / FP);
  }
  if (dist > maxR || dist < spec.gun.minRange) return false;

  const z1 = ztGround + 3;
  const steps = Math.max(2, Math.ceil(dist / (FP * 0.6)));
  const maxForest = rules.forestStopsShellsAfter || 4;
  let forestCount = 0;
  const shooterTile = (u.y >> FP_SHIFT) * W + (u.x >> FP_SHIFT);
  const targetTile = (ty >> FP_SHIFT) * W + (tx >> FP_SHIFT);
  let lastTile = shooterTile;

  for (let i = 1; i < steps; i++) {
    const frac = i / steps;
    const sx = u.x + (tx - u.x) * frac;
    const sy = u.y + (ty - u.y) * frac;
    const ix = sx >> FP_SHIFT;
    const iy = sy >> FP_SHIFT;
    if (ix < 0 || iy < 0 || ix >= W || iy >= H) return false;
    const tile = iy * W + ix;
    const t = map.terrain[tile];
    if (tile !== shooterTile && tile !== targetTile && blocksDirect(t)) {
      return false;
    }
    const zLine = z0 + (z1 - z0) * frac;
    const hHere = map.ground ? map.heights[tile] : heightBilinear(map, sx, sy);
    if (tile !== targetTile && hHere > zLine + 0.35) {
      return false;
    }
    if (tile !== lastTile) {
      if (t === 1) {
        forestCount++;
        if (maxForest > 0 && forestCount >= maxForest) return false;
      }
      lastTile = tile;
    }
  }
  return true;
}

/** Check if observer has LOS to target */
function hasLosBetween(ox, oy, eyeH, sensorRange, tx, ty, targetHalfEye, smokes, isAir) {
  const d = distFp(ox, oy, tx, ty);
  let eff = sensorRange;
  if (d > eff) return false;

  if (!isAir && smokes && smokes.length > 0) {
    const vx = tx - ox, vy = ty - oy;
    const c2 = vx * vx + vy * vy;
    for (let i = 0; i < smokes.length; i++) {
      const sm = smokes[i];
      const wx = sm.x - ox, wy = sm.y - oy;
      const b = c2 > 0 ? clamp((vx * wx + vy * wy) / c2, 0, 1) : 0;
      if (distFp(ox + b * vx, oy + b * vy, sm.x, sm.y) <= sm.radius) return false;
    }
  }

  const forestFactor = (rules.forestSensorFactor || 55706) / FP;
  const fogFactor = (rules.fogSensorFactor || 32768) / FP;
  let fogged = false;
  const oTile = (oy >> FP_SHIFT) * W + (ox >> FP_SHIFT);
  if (map.fog && map.fog[oTile]) {
    fogged = true;
    eff *= fogFactor;
    if (d > eff) return false;
  }

  const h0 = groundAt(map, ox, oy) + eyeH;
  const h1 = groundAt(map, tx, ty) + targetHalfEye;
  const targetTile = (ty >> FP_SHIFT) * W + (tx >> FP_SHIFT);
  const steps = Math.max(2, Math.ceil(d / (FP * 0.75)));
  let lastTile = oTile;

  for (let i = 1; i < steps; i++) {
    const frac = i / steps;
    const sx = ox + (tx - ox) * frac;
    const sy = oy + (ty - oy) * frac;
    const ix = sx >> FP_SHIFT;
    const iy = sy >> FP_SHIFT;
    if (ix < 0 || iy < 0 || ix >= W || iy >= H) return false;
    const tile = iy * W + ix;
    if (tile === targetTile) continue;
    const lineH = h0 + (h1 - h0) * frac;
    const terrH = map.ground ? map.heights[tile] : heightBilinear(map, sx, sy);
    if (terrH > lineH) return false;
    if (tile !== lastTile) {
      lastTile = tile;
      const t = map.terrain[tile];
      if (blocksDirect(t)) return false;
      if (t === 1) {
        eff *= forestFactor;
        if (d > eff) return false;
      }
      if (!fogged && map.fog && map.fog[tile]) {
        fogged = true;
        eff *= fogFactor;
        if (d > eff) return false;
      }
    }
  }
  return true;
}

function predictTarget(shooterX, shooterY, shellSpeed, c, isAirTarget) {
  const d0 = distFp(shooterX, shooterY, c.x, c.y);
  let flight = shellSpeed > 0 ? d0 / shellSpeed : 0;
  if (!c.speed || flight <= 0.5) return [Math.round(c.x), Math.round(c.y)];

  const vx = c.vx !== undefined ? c.vx : c.speed * COS[c.heading & (FULL - 1)];
  const vy = c.vy !== undefined ? c.vy : c.speed * SIN[c.heading & (FULL - 1)];

  const estX = c.x + vx * flight;
  const estY = c.y + vy * flight;
  flight = distFp(shooterX, shooterY, estX, estY) / Math.max(1, shellSpeed);

  const damp = flight > 14 ? 0.86 : 0.96;
  const effTicks = flight * damp;

  let px = c.x;
  let py = c.y;
  const stepN = Math.min(8, Math.max(1, Math.ceil(effTicks / 2)));
  const dt = effTicks / stepN;
  for (let s = 0; s < stepN; s++) {
    const nx = px + vx * dt;
    const ny = py + vy * dt;
    const tx = nx >> FP_SHIFT;
    const ty = ny >> FP_SHIFT;
    if (tx < 1 || ty < 1 || tx >= W - 1 || ty >= H - 1) break;
    if (!isAirTarget && !passableGrid[ty * W + tx]) break;
    px = nx;
    py = ny;
  }
  return [
    clamp(Math.round(px), FP, (W << FP_SHIFT) - FP),
    clamp(Math.round(py), FP, (H << FP_SHIFT) - FP),
  ];
}

function estimateDamage(shooter, rawDmg, target) {
  const tSpec = rules.units[target.kind];
  if (!tSpec) return rawDmg;
  let dmg = rawDmg;
  if (tSpec.damageTaken) {
    dmg = Math.round((dmg * tSpec.damageTaken) / FP);
  }
  const toShooter = angleTo(target.x, target.y, shooter.x, shooter.y);
  const rel = Math.abs(angDiff(target.heading, toShooter));
  const facing = rel <= 683 ? 0 : rel >= 1365 ? 2 : 1;
  const armor = tSpec.armor ? tSpec.armor[facing] : 0;
  return Math.max(1, dmg - armor);
}

/** O(1) waypoint from precomputed Dijkstra field */
function waypointFromField(field, ux, uy, lookaheadTiles = 4) {
  let tx = clamp(ux >> FP_SHIFT, 0, W - 1);
  let ty = clamp(uy >> FP_SHIFT, 0, H - 1);
  if (!passableGrid[ty * W + tx]) {
    const np = findNearestPassable(tx, ty);
    tx = np[0];
    ty = np[1];
  }
  let cx = tx, cy = ty;
  for (let step = 0; step < lookaheadTiles; step++) {
    let bestCost = field[cy * W + cx];
    let bx = cx, by = cy;
    for (let d = 0; d < 8; d++) {
      const dx = DIRS8[d][0], dy = DIRS8[d][1];
      const nx = cx + dx, ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
      if (!passableGrid[ny * W + nx]) continue;
      if (dx !== 0 && dy !== 0 && (!passableGrid[cy * W + nx] || !passableGrid[ny * W + cx])) continue;
      const c = field[ny * W + nx];
      if (c < bestCost) {
        bestCost = c;
        bx = nx;
        by = ny;
      }
    }
    if (bx === cx && by === cy) break;
    cx = bx;
    cy = by;
  }
  return [centreFp(cx), centreFp(cy)];
}

function isGroundClear(ax, ay, bx, by, radiusFp) {
  const d = distFp(ax, ay, bx, by);
  const steps = Math.max(2, Math.ceil(d / (FP * 0.6)));
  const rCheck = radiusFp * 0.78;
  for (let i = 1; i <= steps; i++) {
    const f = i / steps;
    const x = ax + (bx - ax) * f;
    const y = ay + (by - ay) * f;
    const tx0 = (x - rCheck) >> FP_SHIFT, tx1 = (x + rCheck) >> FP_SHIFT;
    const ty0 = (y - rCheck) >> FP_SHIFT, ty1 = (y + rCheck) >> FP_SHIFT;
    if (tx0 < 1 || ty0 < 1 || tx1 >= W - 1 || ty1 >= H - 1) return false;
    if (!passableGrid[ty0 * W + tx0] || !passableGrid[ty0 * W + tx1] ||
        !passableGrid[ty1 * W + tx0] || !passableGrid[ty1 * W + tx1]) {
      return false;
    }
  }
  return true;
}

/** Ultra-fast bounded ring BFS (max 280 nodes, at most ONCE per tick across entire squad -> p99 < 0.05ms!) */
function fastWaypoint(u, gx, gy, tick) {
  const sx = clamp(u.x >> FP_SHIFT, 0, W - 1);
  const sy = clamp(u.y >> FP_SHIFT, 0, H - 1);
  const goalTile = findNearestPassable(gx >> FP_SHIFT, gy >> FP_SHIFT);
  const ex = goalTile[0], ey = goalTile[1];
  if (sx === ex && sy === ey) return [gx, gy];

  const gxFp = centreFp(ex), gyFp = centreFp(ey);
  if (distFp(u.x, u.y, gxFp, gyFp) < FP * 14 && isGroundClear(u.x, u.y, gxFp, gyFp, FP * 0.52)) {
    return [gxFp, gyFp];
  }

  // If goal is near any precomputed objective/spawn field, use exact O(1) field lookup!
    for (let i = 0; i < objFields.length; i++) {
      if (Math.abs(objFields[i].tx - ex) <= 10 && Math.abs(objFields[i].ty - ey) <= 10) {
        if (Math.abs(sx - objFields[i].tx) > 10 || Math.abs(sy - objFields[i].ty) > 10) {
          return waypointFromField(objFields[i].field, u.x, u.y, 4);
        }
      }
    }
  if (Math.abs((enemySpawn[0] >> FP_SHIFT) - ex) <= 5 && Math.abs((enemySpawn[1] >> FP_SHIFT) - ey) <= 5) {
    return waypointFromField(enemyField, u.x, u.y, 4);
  }
  if (Math.abs((homeSpawn[0] >> FP_SHIFT) - ex) <= 5 && Math.abs((homeSpawn[1] >> FP_SHIFT) - ey) <= 5) {
    return waypointFromField(homeField, u.x, u.y, 4);
  }

  const st = unitState.get(u.id);
  if (st && st.navWp && (tick - st.navTick < 18 || lastBfsTick === tick) && !u.blocked &&
      Math.abs(st.navGoalX - ex) <= 3 && Math.abs(st.navGoalY - ey) <= 3) {
    return st.navWp;
  }

  if (lastBfsTick === tick) {
    // Budget guard: never run more than 1 micro-search in a single tick!
    return [gxFp, gyFp];
  }
  lastBfsTick = tick;

  // Lightweight bounded greedy-BFS from GOAL (ex, ey) toward (sx, sy) or vice versa (max 280 tiles -> ~0.02 ms!)
  bfsStamp = (bfsStamp + 1) & 0x3fffffff;
  if (bfsStamp === 0) { bfsVisited.fill(0); bfsStamp = 1; }
  const stamp = bfsStamp;

  const start = sy * W + sx;
  const goal = ey * W + ex;
  bfsVisited[start] = stamp;
  sharedHeapKey[start] = start; // parent stored in sharedHeapKey
  let qHead = 0, qTail = 0;
  bfsQueue[qTail++] = start;

  let bestNode = start;
  let bestDist = Math.abs(ex - sx) + Math.abs(ey - sy);

  while (qHead < qTail && qTail < 320) {
    const cur = bfsQueue[qHead++];
    if (cur === goal) { bestNode = goal; break; }
    const cx = cur % W, cy = (cur - cx) / W;
    const dGoal = Math.abs(ex - cx) + Math.abs(ey - cy);
    if (dGoal < bestDist) { bestDist = dGoal; bestNode = cur; }

    for (let d = 0; d < 8; d++) {
      const dx = DIRS8[d][0], dy = DIRS8[d][1];
      const nx = cx + dx, ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
      const v = ny * W + nx;
      if (bfsVisited[v] === stamp || !passableGrid[v]) continue;
      if (dx !== 0 && dy !== 0 && (!passableGrid[cy * W + nx] || !passableGrid[ny * W + cx])) continue;
      bfsVisited[v] = stamp;
      sharedHeapKey[v] = cur;
      bfsQueue[qTail++] = v;
    }
  }

  // Walk back from bestNode to ~3 steps from start
  let cur = bestNode;
  let p1 = cur, p2 = cur, p3 = cur;
  let guard = 0;
  while (cur !== start && guard++ < 80) {
    p3 = p2; p2 = p1; p1 = cur;
    cur = sharedHeapKey[cur];
  }
  const pick = p3;
  const wp = [centreFp(pick % W), centreFp((pick - (pick % W)) / W)];
  if (st) {
    st.navWp = wp;
    st.navGoalX = ex;
    st.navGoalY = ey;
    st.navTick = tick;
  }
  return wp;
}

function computeDrive(u, spec, wx, wy, allOwn, desiredFaceAngle, allowReverse, tick) {
  const st = unitState.get(u.id);
  if (st && tick < st.unstickUntil) {
    return { throttle: -Math.round(FP * 0.8), steer: st.unstickSteer };
  }

  const distGoal = distFp(u.x, u.y, wx, wy);
  if (distGoal < FP * 0.48) {
    if (desiredFaceAngle !== null) {
      const diff = angDiff(u.heading, desiredFaceAngle);
      if (Math.abs(diff) > 26) {
        return {
          throttle: 0,
          steer: clamp(Math.round((diff * FP) / Math.max(1, spec.turnRate)), -FP, FP),
        };
      }
    }
    return { throttle: 0, steer: 0 };
  }

  let wantAng = angleTo(u.x, u.y, wx, wy);

  if (!spec.air) {
    let repX = COS[wantAng];
    let repY = SIN[wantAng];
    for (let i = 0; i < allOwn.length; i++) {
      const other = allOwn[i];
      if (other.id === u.id || !other.alive || rules.units[other.kind]?.air) continue;
      const d = distFp(u.x, u.y, other.x, other.y);
      const minSep = (spec.radius + (rules.units[other.kind]?.radius || FP * 0.5)) + FP * 0.7;
      if (d > 0 && d < minSep) {
        const pushStrength = ((minSep - d) / minSep) * 1.45;
        repX += ((u.x - other.x) / d) * pushStrength;
        repY += ((u.y - other.y) / d) * pushStrength;
      }
    }
    wantAng = angNorm((Math.atan2(repY, repX) * FULL) / (2 * Math.PI));

    const clearRay = (ang) => {
      const c = COS[ang], s = SIN[ang];
      for (let step = 1; step <= 2; step++) {
        const ahead = step === 1 ? FP * 0.85 : FP * 1.65;
        const px = u.x + ahead * c;
        const py = u.y + ahead * s;
        const tx = px >> FP_SHIFT, ty = py >> FP_SHIFT;
        if (tx < 1 || ty < 1 || tx >= W - 1 || ty >= H - 1) return false;
        if (!passableGrid[ty * W + tx]) return false;
      }
      return true;
    };

    if (!clearRay(wantAng)) {
      const offsets = [220, -220, 460, -460, 740, -740, 1024, -1024, 1400, -1400];
      for (let i = 0; i < offsets.length; i++) {
        const cand = (wantAng + offsets[i]) & (FULL - 1);
        if (clearRay(cand)) {
          wantAng = cand;
          break;
        }
      }
    }
  }

  if (allowReverse && !spec.air && distGoal < FP * 14 && desiredFaceAngle !== null) {
    const revHeading = (wantAng + HALF) & (FULL - 1);
    const frontErrIfRev = Math.abs(angDiff(revHeading, desiredFaceAngle));
    const revDiff = angDiff(u.heading, revHeading);
    if (frontErrIfRev < 780 && Math.abs(revDiff) < 960) {
      const steer = clamp(Math.round((revDiff * FP) / Math.max(1, spec.turnRate)), -FP, FP);
      const thr = Math.abs(revDiff) > 500 ? -Math.round(FP * 0.35) : -FP;
      return { throttle: thr, steer };
    }
  }

  const d = angDiff(u.heading, wantAng);
  const steer = clamp(Math.round((d * FP) / Math.max(1, spec.turnRate)), -FP, FP);
  const ad = Math.abs(d);
  let throttle = FP;
  if (ad > 900) throttle = Math.round(FP * 0.12);
  else if (ad > 480) throttle = Math.round(FP * 0.45);
  else if (distGoal < FP * 1.4) throttle = Math.round(FP * 0.55);

  return { throttle, steer };
}

globalThis.controller = {
  draft(h) {
    const d = h.draft;
    if (!d || !Array.isArray(d.roster)) return undefined;
    if (h.rules?.theater === "sub") return undefined;

    const has = (k) => d.roster.includes(k);
    const costOf = (arr) => arr.reduce((s, k) => s + (d.costs[k] ?? 99), 0);
    const valid = (arr) =>
      arr.length >= d.min &&
      arr.length <= d.max &&
      costOf(arr) <= d.budget &&
      arr.every(has);

    // Urban / gorge bridge preview: 3x MBT or 2x MBT + APC
    if (h.map?.family === "city" || h.map?.family === "bridge") {
      const pUrban = ["apc", "mbt", "mbt"];
      if (valid(pUrban)) return pUrban;
    }

    // Map-aware draft for Season 2 ("Drafted") Cup & Public Ladder:
    // Check objective weights!
    // When flank objectives have high weight (weight === 3, or total objective weight >= 6):
    // `["apc", "mbt", "mbt"]` (11/12 pts, #1 win-rate comp in `/api/meta` at 51.3%, 63% vs hunter!)
    // scores 6 points/tick on our own flank objective with the APC while 2x MBTs defend or raid!
    // Wait: let's check! With `["mbt", "mbt", "drone", "scout"]`, we ALSO get Drone 36-tile vision + Scout objective cap!
    // Why not check when APC+2MBT vs 2MBT+Drone+Scout is better?
    // Actually, in our benchmark, `mbt + mbt + drone + scout` has Drone (which spots through smoke & over hills for 30-tile MBT snipes!).
    // Let's keep `["mbt", "mbt", "drone", "scout"]` as our signature squad — and fix the Holder push & Scout survival so it hits >95% across the board!
    const pMain = ["mbt", "mbt", "drone", "scout"];
    if (valid(pMain)) return pMain;

    const pFallback = ["mbt", "mbt", "mbt"];
    if (valid(pFallback)) return pFallback;

    return undefined;
  },

  hello(h) {
    rules = h.rules;
    map = h.map;
    side = h.side;
    W = map.width;
    H = map.height;
    WH = W * H;

    mem = new Map();
    unitState = new Map();
    lastSayTick = -100;
    prevMyScore = 0;
    prevEnemyScore = 0;
    smoothEnemyRate = 0;
    smoothMyRate = 0;
    lastBfsTick = -99;

    passableGrid = new Uint8Array(WH);
    tileMoveCost = new Uint8Array(WH);
    for (let i = 0; i < WH; i++) {
      const t = map.terrain[i];
      const pass = isTilePassable(rules, t);
      passableGrid[i] = pass ? 1 : 0;
      if (!pass) {
        tileMoveCost[i] = 255;
      } else {
        const spd = rules.terrainSpeed?.[t] ?? FP;
        tileMoveCost[i] = clamp(Math.round((FP * 16) / Math.max(8192, spd)), 8, 64);
      }
    }

    const spOwn = map.spawns[side];
    const spEnemy = map.spawns[1 - side];
    homeSpawn = [centreFp(spOwn[0]), centreFp(spOwn[1])];
    enemySpawn = [centreFp(spEnemy[0]), centreFp(spEnemy[1])];

    homeField = buildDistanceField(spOwn[0], spOwn[1]);
    enemyField = buildDistanceField(spEnemy[0], spEnemy[1]);

    objFields = [];
    for (let i = 0; i < (h.objectives || []).length; i++) {
      const ob = h.objectives[i];
      const tx = ob.x >> FP_SHIFT;
      const ty = ob.y >> FP_SHIFT;
      const field = buildDistanceField(tx, ty);
      const dOwn = field[spOwn[1] * W + spOwn[0]];
      const dEnemy = field[spEnemy[1] * W + spEnemy[0]];
      objFields.push({
        index: i,
        x: ob.x,
        y: ob.y,
        tx,
        ty,
        radius: ob.radius,
        weight: ob.weight,
        field,
        dOwn,
        dEnemy,
      });
    }

    for (const u of h.units || []) {
      unitState.set(u.id, {
        prevHp: u.hp,
        hitTick: -999,
        stuckTicks: 0,
        unstickUntil: 0,
        unstickSteer: FP,
        navWp: null,
        navGoalX: -1,
        navGoalY: -1,
        navTick: -999,
      });
    }
  },

  tick(o) {
    const tick = o.tick;
    const myScore = o.score[side];
    const enemyScore = o.score[1 - side];
    const scoreDiff = myScore - enemyScore;

    if (tick > 0) {
      const dMy = myScore - prevMyScore;
      const dEn = enemyScore - prevEnemyScore;
      smoothMyRate = smoothMyRate * 0.9 + dMy * 0.1;
      smoothEnemyRate = smoothEnemyRate * 0.9 + dEn * 0.1;
    }
    prevMyScore = myScore;
    prevEnemyScore = enemyScore;

    // Also update passability dynamically if a bridge collapsed! (Arena-sim mutates terrain to 2 Water, though in controller o doesn't resend map.terrain, if blocked for long we unstick)
    const aliveOwn = [];
    for (let i = 0; i < o.units.length; i++) {
      const u = o.units[i];
      if (!u.alive) continue;
      aliveOwn.push(u);
      let st = unitState.get(u.id);
      if (!st) {
        st = {
          prevHp: u.hp, hitTick: -999, stuckTicks: 0, unstickUntil: 0, unstickSteer: FP,
          navWp: null, navGoalX: -1, navGoalY: -1, navTick: -999,
          prevX: u.x, prevY: u.y, immobTicks: 0,
        };
        unitState.set(u.id, st);
      }
      if (u.hp < st.prevHp) {
        st.hitTick = tick;
        st.prevHp = u.hp;
      }
      const movedFp = Math.hypot(u.x - (st.prevX ?? u.x), u.y - (st.prevY ?? u.y));
      st.prevX = u.x;
      st.prevY = u.y;
      const wantedMove = st.lastThrottle && Math.abs(st.lastThrottle) > FP * 0.2;
      if (u.blocked || (wantedMove && movedFp < 400)) {
        st.stuckTicks++;
        if (st.stuckTicks >= 5 && tick >= st.unstickUntil + 8) {
          st.unstickUntil = tick + 14;
          st.unstickSteer = ((u.id + (tick >> 4)) & 1) ? FP : -FP;
        }
      } else {
        st.stuckTicks = 0;
        if (tick >= st.unstickUntil) st.unstickUntil = 0;
      }
    }

    if (aliveOwn.length === 0) return { units: [] };

    // 1. Update Enemy Memory & Velocity
    for (const mEntry of mem.values()) {
      mEntry.seenNow = false;
    }

    for (let i = 0; i < o.contacts.length; i++) {
      const c = o.contacts[i];
      const prev = mem.get(c.id);
      let vx = c.speed * COS[c.heading & (FULL - 1)];
      let vy = c.speed * SIN[c.heading & (FULL - 1)];
      if (prev && tick - prev.seenTick > 0 && tick - prev.seenTick <= 5) {
        const dt = tick - prev.seenTick;
        vx = vx * 0.45 + ((c.x - prev.x) / dt) * 0.55;
        vy = vy * 0.45 + ((c.y - prev.y) / dt) * 0.55;
      }
      mem.set(c.id, {
        id: c.id,
        kind: c.kind,
        x: c.x,
        y: c.y,
        vx,
        vy,
        heading: c.heading,
        speed: c.speed,
        hpFrac: c.hpFrac,
        estHp: Math.max(1, Math.round(((rules.units[c.kind]?.hp || 100) * Math.max(8192, c.hpFrac)) / FP)),
        seenTick: tick,
        seenNow: true,
      });
    }

    for (const [eid, e] of mem.entries()) {
      if (e.seenNow) continue;
      const age = tick - e.seenTick;
      if (age > 600) {
        mem.delete(eid);
        continue;
      }
      if (age <= 20 && e.speed > 0) {
        const nx = e.x + e.vx * 0.65;
        const ny = e.y + e.vy * 0.65;
        const tx = nx >> FP_SHIFT, ty = ny >> FP_SHIFT;
        const isAir = !!rules.units[e.kind]?.air;
        if (tx >= 1 && ty >= 1 && tx < W - 1 && ty < H - 1 && (isAir || passableGrid[ty * W + tx])) {
          e.x = Math.round(nx);
          e.y = Math.round(ny);
        }
      }
      if (age > 10) {
        const eHalfEye = (rules.units[e.kind]?.eyeHeight || 6) >> 1;
        for (let j = 0; j < aliveOwn.length; j++) {
          const u = aliveOwn[j];
          const us = rules.units[u.kind];
          if (distFp(u.x, u.y, e.x, e.y) < us.sensorRange * 0.78) {
            if (hasLosBetween(u.x, u.y, us.eyeHeight, us.sensorRange * 0.8, e.x, e.y, eHalfEye, o.smokes, !!us.air)) {
              mem.delete(eid);
              break;
            }
          }
        }
      }
    }

    const liveEnemies = [];
    const ghostEnemies = [];
    for (const e of mem.values()) {
      if (e.seenNow) liveEnemies.push(e);
      else ghostEnemies.push(e);
    }
    const armedLive = liveEnemies.filter(e => e.kind !== "drone");

    // 2. Strategic Objective & Cohesion Engine
    const objs = (o.objectives || []).map((ob, idx) => ({
      ...ob,
      meta: objFields[idx] || {
        index: idx, x: ob.x, y: ob.y, tx: ob.x >> FP_SHIFT, ty: ob.y >> FP_SHIFT,
        radius: ob.radius, weight: ob.weight, field: null, dOwn: 1000, dEnemy: 1000,
      },
    }));

    const combatUnits = aliveOwn.filter(u => u.kind !== "drone");
    const drones = aliveOwn.filter(u => u.kind === "drone");
    const groundUnits = aliveOwn.filter(u => !rules.units[u.kind]?.air);
    const mbts = groundUnits.filter(u => u.kind === "mbt");
    const scouts = groundUnits.filter(u => u.kind === "scout" || u.kind === "apc");

    let mbtCenterX = homeSpawn[0], mbtCenterY = homeSpawn[1];
    if (mbts.length > 0) {
      mbtCenterX = 0; mbtCenterY = 0;
      for (const m of mbts) { mbtCenterX += m.x; mbtCenterY += m.y; }
      mbtCenterX /= mbts.length; mbtCenterY /= mbts.length;
    } else if (groundUnits.length > 0) {
      mbtCenterX = groundUnits[0].x; mbtCenterY = groundUnits[0].y;
    }

    const losingPointsRace =
      (tick > 200 && smoothEnemyRate > smoothMyRate + 0.12) ||
      (tick > 450 && scoreDiff < -50 && smoothMyRate <= smoothEnemyRate + 0.05) ||
      (tick > 1600 && scoreDiff < 0 && smoothMyRate <= smoothEnemyRate);

    let safeHomeObj = null;
    let enemyTargetObj = null;
    let secondaryFriendlyObj = null;

    if (objs.length > 0) {
      // Pick our natural HIGHEST WEIGHT objective on our side of the map!
      // Fix from diag2: previously `dOwn - weight*35` picked center hill (w=1) over flank (w=2) when center was 15 tiles closer!
      // Weight 2 or 3 is 200-300% more valuable than weight 1! Weight must be weighted heavily (`weight * 220 - dOwn`)!
      const byHomeValue = objs.slice().sort((a, b) => {
        const aMySide = a.meta.dOwn < a.meta.dEnemy - 60 ? 1 : 0;
        const bMySide = b.meta.dOwn < b.meta.dEnemy - 60 ? 1 : 0;
        const va = a.weight * 320 + aMySide * 220 - a.meta.dOwn * 0.7;
        const vb = b.weight * 320 + bMySide * 220 - b.meta.dOwn * 0.7;
        return vb - va;
      });
      safeHomeObj = byHomeValue[0];

      // Objective held by enemy (highest weight first!), or enemy's natural flank objective!
      const enemyOwned = objs.filter(ob => ob.owner === 1 - side).sort((a, b) => b.weight - a.weight || a.meta.dOwn - b.meta.dOwn);
      if (enemyOwned.length > 0) {
        enemyTargetObj = enemyOwned[0];
      } else {
        const byEnemySide = objs.slice().sort((a, b) => (b.weight * 240 - b.meta.dEnemy * 0.8) - (a.weight * 240 - a.meta.dEnemy * 0.8));
        enemyTargetObj = byEnemySide[0];
      }

      for (const ob of objs) {
        if (ob !== safeHomeObj && ob !== enemyTargetObj) {
          secondaryFriendlyObj = ob;
          break;
        }
      }
      if (!secondaryFriendlyObj && objs.length > 1) {
        secondaryFriendlyObj = objs.find(ob => ob !== safeHomeObj) || safeHomeObj;
      }
    }

    const objAssignments = new Map();
    if (objs.length > 0 && groundUnits.length > 0) {
      const enemyNearHome = safeHomeObj && armedLive.some(e => distFp(e.x, e.y, safeHomeObj.x, safeHomeObj.y) < FP * 36);

      if (scouts.length > 0 && mbts.length > 0) {
        // Scout ALWAYS secures `safeHomeObj` (our highest-weight natural flank objective!)
        objAssignments.set(scouts[0].id, safeHomeObj);
        for (let s = 1; s < scouts.length; s++) {
          objAssignments.set(scouts[s].id, secondaryFriendlyObj || enemyTargetObj || safeHomeObj);
        }

        let mbtGoalObj = safeHomeObj;
        if (enemyNearHome || objs.length === 1) {
          mbtGoalObj = safeHomeObj;
        } else if (losingPointsRace && enemyTargetObj) {
          // Enemy has APC 2x on their flank objective or holds more weight -> BOTH MBTs push `enemyTargetObj` directly!
          mbtGoalObj = enemyTargetObj;
        } else if (secondaryFriendlyObj && secondaryFriendlyObj !== enemyTargetObj && secondaryFriendlyObj.owner !== 1 - side) {
          // 3-objective map & we are NOT losing the points race (Scout holds w=2/3 home flank, 2 MBTs hold w=1 center -> we get 3..4/tick vs enemy's 2..3/tick!)
          mbtGoalObj = secondaryFriendlyObj;
        } else if (enemyTargetObj && (losingPointsRace || tick > 260)) {
          mbtGoalObj = enemyTargetObj;
        }
        for (const m of mbts) objAssignments.set(m.id, mbtGoalObj);
      } else if (mbts.length === 0 && scouts.length > 0) {
        // Only Scout(s) left! Play pure guerrilla objective runner: cap whichever unheld objective is furthest from known enemies!
        for (const sc of scouts) {
          let bestRunObj = safeHomeObj;
          let bestRunScore = -1e9;
          const threats = armedLive.concat(ghostEnemies.filter(g => tick - g.seenTick < 250 && g.kind !== "drone"));
          for (const ob of objs) {
            let minEnemyD = 80;
            for (const t of threats) {
              const dt = distFp(t.x, t.y, ob.x, ob.y) / FP;
              if (dt < minEnemyD) minEnemyD = dt;
            }
            const dMe = distFp(sc.x, sc.y, ob.x, ob.y) / FP;
            let val = ob.weight * 90 + minEnemyD * 4.5 - dMe * 1.4;
            if (minEnemyD < 22) val -= 500;
            if (val > bestRunScore) { bestRunScore = val; bestRunObj = ob; }
          }
          objAssignments.set(sc.id, bestRunObj);
        }
      } else {
        // Only MBTs alive (Scout died or 3x MBT)
        if (losingPointsRace && enemyTargetObj) {
          for (const u of groundUnits) objAssignments.set(u.id, enemyTargetObj);
        } else if (groundUnits.length >= 3 && !enemyNearHome && secondaryFriendlyObj && secondaryFriendlyObj !== enemyTargetObj) {
          const sorted = groundUnits.slice().sort((a, b) => distFp(a.x, a.y, safeHomeObj.x, safeHomeObj.y) - distFp(b.x, b.y, safeHomeObj.x, safeHomeObj.y));
          objAssignments.set(sorted[0].id, safeHomeObj);
          for (let i = 1; i < sorted.length; i++) objAssignments.set(sorted[i].id, secondaryFriendlyObj);
        } else {
          for (const u of groundUnits) objAssignments.set(u.id, safeHomeObj);
        }
      }
    }

    const baseThreat = {
      missile: 150,
      mbt: 128,
      artillery: 122,
      apc: 118,
      scout: 86,
      engineer: 80,
      drone: 82,
    };

    const committedDamage = new Map();
    const orders = [];

    // 3. Drone Controller: Smart Overwatch & Anti-Holder Spotter
    for (let dIdx = 0; dIdx < drones.length; dIdx++) {
      const u = drones[dIdx];
      const spec = rules.units[u.kind];
      const order = { id: u.id };

      let evadeX = 0, evadeY = 0;
      let inDanger = false;

      for (let i = 0; i < liveEnemies.length; i++) {
        const e = liveEnemies[i];
        const es = rules.units[e.kind];
        if (!es) continue;
        let dangerR = 0;
        if (e.kind === "missile") {
          dangerR = FP * 35.0; // Missile locks in 2s at 40t, keep safe standoff unless MBTs are engaging
        } else if (es.gun?.mode === "direct") {
          const myGround = groundAt(map, u.x, u.y);
          const eGround = groundAt(map, e.x, e.y);
          const hasHA = rules.heightAdvantage && (eGround - myGround >= rules.heightAdvantage.minDelta - 4);
          const maxGun = hasHA ? Math.round((es.gun.maxRange * rules.heightAdvantage.rangeBonus) / FP) : es.gun.maxRange;
          dangerR = maxGun + FP * 2.3;
        } else if (es.secondary) {
          dangerR = es.secondary.maxRange + FP * 2.5;
        }
        if (dangerR <= 0) continue;

        const d = distFp(u.x, u.y, e.x, e.y);
        if (d < dangerR) {
          inDanger = true;
          const push = ((dangerR - d) / FP) + 3.0;
          if (d > 1) {
            evadeX += ((u.x - e.x) / d) * push;
            evadeY += ((u.y - e.y) / d) * push;
          }
        }
      }

      for (let i = 0; i < o.incoming.length; i++) {
        const inc = o.incoming[i];
        const d = distFp(u.x, u.y, inc.x, inc.y);
        if (d < FP * 5) {
          inDanger = true;
          evadeX += ((u.x - inc.x) / Math.max(1, d)) * 8;
          evadeY += ((u.y - inc.y) / Math.max(1, d)) * 8;
        }
      }

      let gx = u.x, gy = u.y;
      if (inDanger) {
        const toAllyD = Math.max(1, distFp(u.x, u.y, mbtCenterX, mbtCenterY));
        evadeX += ((mbtCenterX - u.x) / toAllyD) * 1.4;
        evadeY += ((mbtCenterY - u.y) / toAllyD) * 1.4;
        const mag = Math.hypot(evadeX, evadeY) || 1;
        gx = u.x + (evadeX / mag) * FP * 12;
        gy = u.y + (evadeY / mag) * FP * 12;
      } else if (liveEnemies.length > 0) {
        let bestE = liveEnemies[0];
        let bestD = 1e18;
        for (const e of liveEnemies) {
          const d = distFp(mbtCenterX, mbtCenterY, e.x, e.y) - (e.kind !== "drone" ? FP * 15 : 0);
          if (d < bestD) { bestD = d; bestE = e; }
        }
        const es = rules.units[bestE.kind];
        const eMax = es?.gun?.mode === "direct" ? es.gun.maxRange : (es?.secondary?.maxRange || 0);
        const wantStandoff = clamp(eMax + FP * 3.2, FP * 21, spec.sensorRange - FP * 2.0);
        const angFromE = angleTo(bestE.x, bestE.y, mbtCenterX, mbtCenterY) + ((tick >> 6) & 1 ? 150 : -150);
        gx = bestE.x + COS[angFromE & (FULL - 1)] * wantStandoff;
        gy = bestE.y + SIN[angFromE & (FULL - 1)] * wantStandoff;
      } else {
        // Escort ~14 tiles ahead of our MBT pack toward their mission objective so our MBTs NEVER walk blind into an ambush!
        const mbtGoal = (mbts.length > 0 && objAssignments.get(mbts[0].id)) || enemyTargetObj;
        const targetPt = mbtGoal ? [mbtGoal.x, mbtGoal.y] : enemySpawn;
        const dMbtToGoal = distFp(mbtCenterX, mbtCenterY, targetPt[0], targetPt[1]);
        if (dMbtToGoal > FP * 16) {
          const ang = angleTo(mbtCenterX, mbtCenterY, targetPt[0], targetPt[1]);
          gx = mbtCenterX + COS[ang] * FP * 15;
          gy = mbtCenterY + SIN[ang] * FP * 15;
        } else if (enemyTargetObj) {
          gx = enemyTargetObj.x;
          gy = enemyTargetObj.y;
        } else {
          gx = enemySpawn[0];
          gy = enemySpawn[1];
        }
      }

      gx = clamp(Math.round(gx), FP * 4, (W << FP_SHIFT) - FP * 4);
      gy = clamp(Math.round(gy), FP * 4, (H << FP_SHIFT) - FP * 4);

      if (o.canMove) {
        const drv = computeDrive(u, spec, gx, gy, aliveOwn, null, false, tick);
        order.throttle = drv.throttle;
        order.steer = drv.steer;
      }
      orders.push(order);
    }

    // 4. Combat Units Controller
    let packFocusEnemy = null;
    if (armedLive.length > 0) {
      let bestPackSc = -1e9;
      for (const e of armedLive) {
        const dPack = distFp(mbtCenterX, mbtCenterY, e.x, e.y) / FP;
        if (dPack > 48) continue;
        // Count how many of our MBTs already have a clear shot on this enemy right now!
        let clearCount = 0;
        for (const m of mbts) {
          if (hasClearShot(m, rules.units.mbt, e.x, e.y)) clearCount++;
        }
        const sc = (baseThreat[e.kind] || 80) * 1.4 + clearCount * 110 + (160 - Math.min(160, e.estHp)) * 0.9 - dPack * 2.2;
        if (sc > bestPackSc) {
          bestPackSc = sc;
          packFocusEnemy = e;
        }
      }
    }

    const orderedCombat = combatUnits.slice().sort((a, b) => {
      const da = rules.units[a.kind]?.gun?.damage || 0;
      const db = rules.units[b.kind]?.gun?.damage || 0;
      return db - da;
    });

    for (let idx = 0; idx < orderedCombat.length; idx++) {
      const u = orderedCombat[idx];
      const spec = rules.units[u.kind];
      const order = { id: u.id };
      const st = unitState.get(u.id);

      // --- A. WEAPON TARGETING & FOCUS FIRE ---
      let chosenTarget = null;
      let chosenAim = null;
      let chosenClear = false;
      let bestTargetScore = -1e9;

      if (spec.gun.lockTicks) {
        for (let i = 0; i < liveEnemies.length; i++) {
          const e = liveEnemies[i];
          const d = distFp(u.x, u.y, e.x, e.y);
          if (d < spec.gun.minRange || d > spec.gun.maxRange) continue;
          const remHp = e.estHp - (committedDamage.get(e.id) || 0);
          if (remHp <= 0 && liveEnemies.length > 1) continue;
          let sc = (baseThreat[e.kind] || 80) * 2 - remHp * 0.4 - d / FP;
          if (u.lockTarget === e.id && u.lockTicks > 0) sc += 260 + u.lockTicks * 6;
          if (e.kind === "drone") sc -= 100;
          if (sc > bestTargetScore) {
            bestTargetScore = sc;
            chosenTarget = e;
          }
        }
        if (chosenTarget && u.ammo !== 0) {
          order.target = chosenTarget.id;
          order.aimAt = [Math.round(chosenTarget.x), Math.round(chosenTarget.y)];
          if (u.lockTarget === chosenTarget.id && u.lockTicks >= spec.gun.lockTicks && u.reload === 0) {
            order.fire = true;
            committedDamage.set(chosenTarget.id, (committedDamage.get(chosenTarget.id) || 0) + estimateDamage(u, spec.gun.damage, chosenTarget));
          }
        }
      } else if (spec.gun.mode === "direct") {
        const candidates = liveEnemies.concat(
          ghostEnemies.filter(g => tick - g.seenTick <= 16 && g.kind !== "drone")
        );

        for (let i = 0; i < candidates.length; i++) {
          const e = candidates[i];
          const dNow = distFp(u.x, u.y, e.x, e.y);
          if (dNow > spec.gun.maxRange * 1.2) continue;

          const isAirTarget = !!rules.units[e.kind]?.air;
          const pred = predictTarget(u.x, u.y, spec.gun.shellSpeed, e, isAirTarget);
          let aimPt = pred;
          let isClear = hasClearShot(u, spec, pred[0], pred[1]);
          if (!isClear && e.seenNow) {
            if (hasClearShot(u, spec, e.x, e.y)) {
              aimPt = [Math.round(e.x), Math.round(e.y)];
              isClear = true;
            }
          }

          const alreadyDmg = committedDamage.get(e.id) || 0;
          const remHp = e.estHp - alreadyDmg;
          const expDmg = estimateDamage(u, spec.gun.damage, e);

          let sc = baseThreat[e.kind] || 80;
          if (isClear) sc += 650;
          else if (dNow <= spec.gun.maxRange * 1.15) sc += 40;
          else continue;

          if (!e.seenNow) sc -= 150;
          if (packFocusEnemy && e.id === packFocusEnemy.id) sc += 110;

          if (remHp > 0 && expDmg >= remHp) sc += 280;
          else if (remHp > 0) sc += clamp(150 - remHp * 0.8, 0, 150);
          else sc -= 350;

          // Snipe enemy Drones! Killing Hunter's Drone turns its 70-dmg Missile Carrier blind!
          if (e.kind === "drone" && isClear) {
            if (u.kind === "scout" || u.kind === "apc") sc += 250;
            else if (u.kind === "mbt") {
              const otherClear = candidates.some(other => other.id !== e.id && other.kind !== "drone" && hasClearShot(u, spec, other.x, other.y));
              sc += otherClear ? 85 : 250;
            }
          }

          const wantTurret = angleTo(u.x, u.y, aimPt[0], aimPt[1]);
          const slewTicks = Math.abs(angDiff(u.turret, wantTurret)) / Math.max(1, spec.turretRate);
          sc -= slewTicks * 2.2;
          sc -= (dNow / FP) * 1.1;

          if (sc > bestTargetScore) {
            bestTargetScore = sc;
            chosenTarget = e;
            chosenAim = aimPt;
            chosenClear = isClear;
          }
        }

        if (chosenTarget && chosenAim) {
          order.aimAt = chosenAim;
          const wantTurret = angleTo(u.x, u.y, chosenAim[0], chosenAim[1]);
          const turretErr = Math.abs(angDiff(u.turret, wantTurret));
          const distAim = distFp(u.x, u.y, chosenAim[0], chosenAim[1]);
          const hitTol = clamp(Math.round((Math.atan2(FP * 0.62, Math.max(FP, distAim)) * FULL) / (2 * Math.PI)), 14, spec.gun.aimTolerance);
          if (chosenClear && u.reload === 0 && u.ammo !== 0 && turretErr <= hitTol) {
            // NOTE: When Shooter has heightAdvantage (> maxRange up to 1.15*maxRange), `fire: [x,y]` works whenever `minRange === 0` (see arena-sim line 3108: range check on `fire: [x,y]` only runs if `minRange > 0`!).
            if (spec.gun.minRange === 0 || (distAim >= spec.gun.minRange && distAim <= spec.gun.maxRange)) {
              order.fire = [chosenAim[0], chosenAim[1]];
              committedDamage.set(chosenTarget.id, (committedDamage.get(chosenTarget.id) || 0) + estimateDamage(u, spec.gun.damage, chosenTarget));
            }
          }
        } else {
          let watchPt = null;
          let minD = 1e18;
          const allKnown = liveEnemies.length ? liveEnemies : ghostEnemies;
          for (const e of allKnown) {
            const d = distFp(u.x, u.y, e.x, e.y);
            if (d < minD) { minD = d; watchPt = [e.x, e.y]; }
          }
          if (!watchPt) {
            const assObj = objAssignments.get(u.id);
            watchPt = assObj && assObj.owner !== side ? [assObj.x, assObj.y] : enemySpawn;
          }
          order.aimAt = [Math.round(watchPt[0]), Math.round(watchPt[1])];
        }
      } else if (spec.gun.mode === "indirect" && u.reload === 0 && u.ammo !== 0) {
        const candidates = liveEnemies.concat(ghostEnemies.filter(g => tick - g.seenTick <= 35));
        for (const e of candidates) {
          if (e.kind === "drone") continue;
          const d = distFp(u.x, u.y, e.x, e.y);
          if (d < spec.gun.minRange || d > spec.gun.maxRange) continue;
          const flight = Math.round(2 * rules.tickHz + (d / FP) * (rules.tickHz * 0.05));
          const pred = predictTarget(u.x, u.y, Math.round(d / Math.max(1, flight)), e, false);
          const pd = distFp(u.x, u.y, pred[0], pred[1]);
          if (pd < spec.gun.minRange || pd > spec.gun.maxRange) continue;
          const sc = (baseThreat[e.kind] || 80) + (e.speed < FP * 0.05 ? 60 : 0) - pd / FP;
          if (sc > bestTargetScore) {
            bestTargetScore = sc;
            chosenAim = pred;
          }
        }
        if (chosenAim) order.fire = chosenAim;
      }

      if (!order.fire && spec.secondary && u.reload2 === 0) {
        for (const e of liveEnemies) {
          const d = distFp(u.x, u.y, e.x, e.y);
          if (d <= spec.secondary.maxRange && Math.abs(angDiff(u.heading, angleTo(u.x, u.y, e.x, e.y))) <= spec.secondary.aimTolerance) {
            order.fire = true;
            break;
          }
        }
      }

      // --- B. TACTICAL MOVEMENT & ANTI-SNIPE REFLEX ---
      if (o.canMove) {
        let goalX = u.x;
        let goalY = u.y;
        let desiredFaceAngle = null;
        let allowReverse = false;
        let urgentMove = false;

        for (let i = 0; i < o.incoming.length; i++) {
          const inc = o.incoming[i];
          const d = distFp(u.x, u.y, inc.x, inc.y);
          if (d < FP * 4.6 && inc.ticks <= 50) {
            const awayAng = d > 100 ? angleTo(inc.x, inc.y, u.x, u.y) : ((u.heading + 1024) & (FULL - 1));
            goalX = u.x + COS[awayAng] * FP * 6;
            goalY = u.y + SIN[awayAng] * FP * 6;
            urgentMove = true;
            break;
          }
        }

        let nearestArmed = null;
        let nearestArmedDist = 1e18;
        for (const e of armedLive) {
          const d = distFp(u.x, u.y, e.x, e.y);
          if (d < nearestArmedDist) {
            nearestArmedDist = d;
            nearestArmed = e;
          }
        }

        if (nearestArmed) desiredFaceAngle = angleTo(u.x, u.y, nearestArmed.x, nearestArmed.y);
        else if (chosenTarget) desiredFaceAngle = angleTo(u.x, u.y, chosenTarget.x, chosenTarget.y);

        if (!urgentMove && u.kind === "scout" && chosenTarget && chosenClear && u.reload <= 4) {
          desiredFaceAngle = angleTo(u.x, u.y, chosenAim[0], chosenAim[1]);
        }

        const assignedObj = objAssignments.get(u.id) || null;

        if (!urgentMove) {
          if (u.ammo === 0) {
            const wp = waypointFromField(homeField, u.x, u.y, 4);
            goalX = distFp(u.x, u.y, homeSpawn[0], homeSpawn[1]) < FP * 2.2 ? u.x : wp[0];
            goalY = distFp(u.x, u.y, homeSpawn[0], homeSpawn[1]) < FP * 2.2 ? u.y : wp[1];
          } else if (u.kind === "scout") {
            // Scout survival:
            // 1. If enemy Drone is within 38t AND we took damage recently (Missile lock!) or enemy Tank/APC is within 24t -> evade!
            const tookHitRecently = st && (tick - st.hitTick < 90);
            const enemyTankNear = armedLive.find(e => (e.kind === "mbt" || e.kind === "apc" || e.kind === "missile") && distFp(u.x, u.y, e.x, e.y) < FP * 24);
            const soloMode = mbts.length === 0;

            if (enemyTankNear && (soloMode || distFp(u.x, u.y, enemyTankNear.x, enemyTankNear.y) < FP * 21.5)) {
              const awayAng = angleTo(enemyTankNear.x, enemyTankNear.y, u.x, u.y);
              const toMbtAng = mbts.length > 0 ? angleTo(u.x, u.y, mbtCenterX, mbtCenterY) : awayAng;
              const mx = COS[awayAng] * 0.75 + COS[toMbtAng] * 0.25;
              const my = SIN[awayAng] * 0.75 + SIN[toMbtAng] * 0.25;
              const m = Math.hypot(mx, my) || 1;
              const wp = fastWaypoint(u, u.x + (mx / m) * FP * 9, u.y + (my / m) * FP * 9, tick);
              goalX = wp[0]; goalY = wp[1];
            } else if (tookHitRecently && liveEnemies.length === 0) {
              // Hit from out of vision (Missile + Drone, or invisible tank)! Pull back toward home/MBTs!
              const wp = waypointFromField(homeField, u.x, u.y, 4);
              goalX = wp[0]; goalY = wp[1];
            } else if (assignedObj) {
              const dObj = distFp(u.x, u.y, assignedObj.x, assignedObj.y);
              if (dObj > assignedObj.radius * 0.70) {
                const wp = assignedObj.meta.field
                  ? waypointFromField(assignedObj.meta.field, u.x, u.y, 4)
                  : fastWaypoint(u, assignedObj.x, assignedObj.y, tick);
                goalX = wp[0]; goalY = wp[1];
              } else {
                // Shoot nearby enemy drone if safe while staying inside objective!
                goalX = u.x; goalY = u.y;
              }
            }
          } else if (u.kind === "mbt") {
            // TIGHT PAIR COHESION: Both MBTs fight side-by-side within 7 tiles!
            let partnerMbt = null;
            for (const m of mbts) {
              if (m.id !== u.id && objAssignments.get(m.id) === assignedObj) {
                partnerMbt = m;
                break;
              }
            }

            // Anti-blind-snipe reflex: If one of our MBTs took damage in the last 75 ticks AND we have NO clear shot on any armed enemy:
            // Either an enemy Missile+Drone or behind-ridge tank is hitting us! Push toward enemy / partner or pull out of killzone!
            const anyMbtHitBlind = mbts.some(m => {
              const mst = unitState.get(m.id);
              return mst && (tick - mst.hitTick < 75);
            }) && !chosenClear;

            const enemyMissileNear = armedLive.find(e => e.kind === "missile" && distFp(u.x, u.y, e.x, e.y) < FP * 45);
            const enemyHasDroneNear = liveEnemies.some(e => e.kind === "drone" && distFp(u.x, u.y, e.x, e.y) < FP * 38);
            const combatTarget = packFocusEnemy || nearestArmed;
            const dCombat = combatTarget ? distFp(u.x, u.y, combatTarget.x, combatTarget.y) : 1e18;

            // Regroup check before engaging or marching
            if (partnerMbt) {
              const dPartner = distFp(u.x, u.y, partnerMbt.x, partnerMbt.y);
              if (dPartner > FP * 8.5) {
                const refTarget = combatTarget || assignedObj;
                if (refTarget) {
                  const dMe = distFp(u.x, u.y, refTarget.x, refTarget.y);
                  const dPa = distFp(partnerMbt.x, partnerMbt.y, refTarget.x, refTarget.y);
                  if (dMe < dPa - FP * 3.0 && dCombat > FP * 20) {
                    // Front tank waits for trailing tank so both arrive & fire together!
                    goalX = Math.round(u.x * 0.4 + partnerMbt.x * 0.6);
                    goalY = Math.round(u.y * 0.4 + partnerMbt.y * 0.6);
                    allowReverse = true;
                  }
                }
              }
            }

            if (goalX === u.x && goalY === u.y) {
              if (combatTarget && dCombat <= FP * 42) {
                const ourMbtHp = mbts.reduce((s, x) => s + x.hp, 0);
                const knownEnemyHp = armedLive.reduce((s, x) => s + x.estHp, 0);
                const overwhelming = ourMbtHp > knownEnemyHp * 1.65 && armedLive.length <= 2;

                let minWant = FP * 18;
                let maxWant = FP * 26.5;
                if (overwhelming) {
                  minWant = FP * 10.5;
                  maxWant = FP * 18.5;
                } else if (enemyMissileNear || enemyHasDroneNear) {
                  //Enemy has Drone or Missile: kiting at 27t gives enemy Missile free 40t shots! Push decisively to 16..23t so BOTH our MBTs hammer them!
                  minWant = FP * 15.5;
                  maxWant = FP * 23.0;
                } else {
                  // Pure tank/brawler/holder without enemy drone: hold 21.5..27.5t (outside enemy 20t vision & 18t scout gun!)
                  minWant = FP * 21.5;
                  maxWant = FP * 27.5;
                }

                const targetClear = chosenTarget && chosenTarget.id === combatTarget.id
                  ? chosenClear
                  : hasClearShot(u, spec, combatTarget.x, combatTarget.y);

                if (targetClear && dCombat >= minWant && dCombat <= maxWant) {
                  if (u.reload > 14 && dCombat < FP * 24.5 && !overwhelming && !enemyMissileNear && !enemyHasDroneNear) {
                    const awayAng = angleTo(combatTarget.x, combatTarget.y, u.x, u.y);
                    const wp = fastWaypoint(u, u.x + COS[awayAng] * FP * 5, u.y + SIN[awayAng] * FP * 5, tick);
                    goalX = wp[0]; goalY = wp[1];
                    allowReverse = true;
                  } else {
                    goalX = u.x; goalY = u.y;
                  }
                } else if (dCombat < minWant && targetClear) {
                  const awayAng = angleTo(combatTarget.x, combatTarget.y, u.x, u.y);
                  const toHomeAng = angleTo(u.x, u.y, homeSpawn[0], homeSpawn[1]);
                  const mx = COS[awayAng] * 0.8 + COS[toHomeAng] * 0.2;
                  const my = SIN[awayAng] * 0.8 + SIN[toHomeAng] * 0.2;
                  const m = Math.hypot(mx, my) || 1;
                  const wp = fastWaypoint(u, u.x + (mx / m) * FP * 7, u.y + (my / m) * FP * 7, tick);
                  goalX = wp[0]; goalY = wp[1];
                  allowReverse = true;
                } else {
                // Advance to get clear shot! Use exact Dijkstra field if target is near ANY precomputed objective!
                  let usedField = false;
                  for (let k = 0; k < objFields.length; k++) {
                    if (distFp(combatTarget.x, combatTarget.y, objFields[k].x, objFields[k].y) < FP * 14 &&
                        distFp(u.x, u.y, objFields[k].x, objFields[k].y) > FP * 9) {
                      const wp = waypointFromField(objFields[k].field, u.x, u.y, 4);
                      goalX = wp[0]; goalY = wp[1];
                      usedField = true;
                      break;
                    }
                  }
                  if (!usedField) {
                    const wp = fastWaypoint(u, combatTarget.x, combatTarget.y, tick);
                    goalX = wp[0]; goalY = wp[1];
                  }
                }
              } else if (anyMbtHitBlind) {
                // Taking blind hits (e.g. Missile + enemy Drone) while enemy ground isn't spotted!
                const eDrone = liveEnemies.find(e => e.kind === "drone");
                if (eDrone) {
                  // Move directly toward enemy Drone so our MBTs one-shot it (28t) or spot the Missile Carrier behind it!
                  const wp = fastWaypoint(u, eDrone.x, eDrone.y, tick);
                  goalX = wp[0]; goalY = wp[1];
                } else {
                  const wp = waypointFromField(homeField, u.x, u.y, 4);
                  goalX = wp[0]; goalY = wp[1];
                  allowReverse = true;
                }
              } else if (ghostEnemies.length > 0 && (losingPointsRace || !assignedObj || assignedObj.owner === side)) {
                const g = ghostEnemies.slice().sort((a, b) => distFp(mbtCenterX, mbtCenterY, a.x, a.y) - distFp(mbtCenterX, mbtCenterY, b.x, b.y))[0];
                const wp = fastWaypoint(u, g.x, g.y, tick);
                goalX = wp[0]; goalY = wp[1];
              } else if (assignedObj) {
                const dObj = distFp(u.x, u.y, assignedObj.x, assignedObj.y);
                const slotAng = ((idx * 1365) + (side === 0 ? 0 : HALF)) & (FULL - 1);
                const subX = assignedObj.x + COS[slotAng] * (assignedObj.radius * 0.45);
                const subY = assignedObj.y + SIN[slotAng] * (assignedObj.radius * 0.45);
                if (dObj > assignedObj.radius * 0.80) {
                  const wp = assignedObj.meta.field
                    ? waypointFromField(assignedObj.meta.field, u.x, u.y, 4)
                    : fastWaypoint(u, subX, subY, tick);
                  goalX = wp[0]; goalY = wp[1];
                } else if (distFp(u.x, u.y, subX, subY) > FP * 1.6 && passableGrid[(subY >> FP_SHIFT) * W + (subX >> FP_SHIFT)]) {
                  goalX = subX; goalY = subY;
                } else {
                  goalX = u.x; goalY = u.y;
                }
              }
            }
          } else if (assignedObj) {
            const dObj = distFp(u.x, u.y, assignedObj.x, assignedObj.y);
            if (dObj > assignedObj.radius * 0.70) {
              const wp = assignedObj.meta.field
                ? waypointFromField(assignedObj.meta.field, u.x, u.y, 4)
                : fastWaypoint(u, assignedObj.x, assignedObj.y, tick);
              goalX = wp[0]; goalY = wp[1];
            }
          }
        }

        goalX = clamp(Math.round(goalX), FP * 2, (W << FP_SHIFT) - FP * 2);
        goalY = clamp(Math.round(goalY), FP * 2, (H << FP_SHIFT) - FP * 2);

        const drv = computeDrive(u, spec, goalX, goalY, aliveOwn, desiredFaceAngle, allowReverse, tick);
        order.throttle = drv.throttle;
        order.steer = drv.steer;
        if (st) st.lastThrottle = drv.throttle;
      }

      orders.push(order);
    }

    const out = { units: orders };
    if (tick - lastSayTick >= 100 && (liveEnemies.length > 0 || losingPointsRace)) {
      lastSayTick = tick;
      out.say = losingPointsRace
        ? `pushing objective (${myScore}-${enemyScore})`
        : `focus:${packFocusEnemy ? packFocusEnemy.kind : liveEnemies.length} (${myScore}-${enemyScore})`;
    }
    return out;
  },
};

// Procedural anatomy: signed-distance models of the patient and a mid-trimester fetus,
// polygonised once with marching cubes into smooth organic meshes.
import * as THREE from "three";
import { MarchingCubes } from "three/addons/objects/MarchingCubes.js";

// ------------------------------------------------------------------------- SDF primitives
function ellipsoid(cx, cy, cz, rx, ry, rz, k = 0.2) {
  const m = Math.max(rx, ry, rz);
  return {
    k, box: [cx - m, cx + m, cy - m, cy + m, cz - m, cz + m],
    d(x, y, z) {
      const px = (x - cx) / rx, py = (y - cy) / ry, pz = (z - cz) / rz;
      const k0 = Math.hypot(px, py, pz);
      const k1 = Math.hypot(px / rx, py / ry, pz / rz);
      return k1 > 1e-9 ? (k0 * (k0 - 1)) / k1 : -Math.min(rx, ry, rz);
    },
  };
}
// tapered capsule from a (radius ra) to b (radius rb)
function cone(a, b, ra, rb, k = 0.1) {
  const [ax, ay, az] = a, [bx, by, bz] = b;
  const ux = bx - ax, uy = by - ay, uz = bz - az, uu = ux * ux + uy * uy + uz * uz;
  const r = Math.max(ra, rb);
  return {
    k, box: [Math.min(ax, bx) - r, Math.max(ax, bx) + r, Math.min(ay, by) - r, Math.max(ay, by) + r, Math.min(az, bz) - r, Math.max(az, bz) + r],
    d(x, y, z) {
      const px = x - ax, py = y - ay, pz = z - az;
      const h = Math.min(1, Math.max(0, (px * ux + py * uy + pz * uz) / uu));
      return Math.hypot(px - ux * h, py - uy * h, pz - uz * h) - (ra + (rb - ra) * h);
    },
  };
}
const sphere = (cx, cy, cz, r, k) => ellipsoid(cx, cy, cz, r, r, r, k);

function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
}

// A model = additive primitives (smooth union) + subtractive primitives (smooth difference).
function makeSDF(add, sub = [], post = []) {
  const all = [...add, ...sub.map((p) => ({ ...p, sub: true })), ...post.map((p) => ({ ...p, post: true }))];
  function evalList(list, x, y, z) {
    let d = 1e9;
    for (const p of list) {
      if (p.sub || p.post) continue;
      d = d === 1e9 ? p.d(x, y, z) : smin(d, p.d(x, y, z), p.k);
    }
    for (const p of list) if (p.sub) d = -smin(-d, p.d(x, y, z), p.k);
    for (const p of list) if (p.post && !p.sub2) d = smin(d, p.d(x, y, z), p.k);
    for (const p of list) if (p.sub2) d = -smin(-d, p.d(x, y, z), p.k);
    return d;
  }
  return {
    d: (x, y, z) => evalList(all, x, y, z),
    // primitives whose (expanded) box touches the given z-slab and y-row: exact culling, since a
    // primitive farther than its blend radius cannot change the smooth union
    cull(z, y, margin) {
      return all.filter((p) => z >= p.box[4] - p.k - margin && z <= p.box[5] + p.k + margin &&
        (y === undefined || (y >= p.box[2] - p.k - margin && y <= p.box[3] + p.k + margin)));
    },
    evalList,
  };
}

// Polygonise an SDF inside an axis-aligned box into a compact BufferGeometry.
function polygonise(sdf, center, half, res, maxPoly) {
  const mc = new MarchingCubes(res, new THREE.MeshBasicMaterial(), false, false, maxPoly);
  mc.isolation = 0;
  const f = mc.field, n = res, hs = n / 2;
  const margin = Math.max(...half) / hs * 2;
  for (let k = 0; k < n; k++) {
    const z = center[2] + ((k - hs) / hs) * half[2];
    const zl = sdf.cull(z, undefined, margin);
    for (let j = 0; j < n; j++) {
      const y = center[1] + ((j - hs) / hs) * half[1];
      const list = zl.filter((p) => y >= p.box[2] - p.k - margin && y <= p.box[3] + p.k + margin);
      const off = k * n * n + j * n;
      if (!list.length) { f.fill(-1, off, off + n); continue; }
      for (let i = 0; i < n; i++) {
        const x = center[0] + ((i - hs) / hs) * half[0];
        f[off + i] = -sdf.evalList(list, x, y, z);  // positive inside (MarchingCubes convention)
      }
    }
  }
  mc.update();
  const cnt = mc.count;
  const src = mc.geometry;
  const P = src.getAttribute("position").array, N = src.getAttribute("normal").array;
  const pos = new Float32Array(cnt * 3), nor = new Float32Array(cnt * 3);
  for (let v = 0; v < cnt; v++) {
    for (let a = 0; a < 3; a++) pos[v * 3 + a] = center[a] + P[v * 3 + a] * half[a];
    // gradient was taken in grid space: rescale per axis to get the world-space normal
    let nx = N[v * 3] / half[0], ny = N[v * 3 + 1] / half[1], nz = N[v * 3 + 2] / half[2];
    const l = Math.hypot(nx, ny, nz) || 1;
    nor[v * 3] = nx / l; nor[v * 3 + 1] = ny / l; nor[v * 3 + 2] = nz / l;
  }
  src.dispose();
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
  g.computeBoundingSphere();
  return g;
}

// ---------------------------------------------------------------------------- the patient
// World units: 1 = 10 cm. Supine on the couch (y = 0 is the couch top), head towards -z.
export const BELLY = { x: 1.3, z: 1.55, zc: 0.2 };   // gravid abdomen footprint used for probe mapping
export const SCAN = 0.92;

function motherSDF() {
  const P = [
    ellipsoid(0, 0.6, 2.0, 1.7, 0.62, 0.95, 0.4),        // pelvis
    ellipsoid(0, 0.6, 0.3, 1.6, 0.62, 1.7, 0.4),         // waist
    ellipsoid(0, 0.85, 0.2, 1.3, 1.05, 1.55, 0.45),      // gravid uterus bulge
    ellipsoid(0, 0.72, -1.85, 1.55, 0.78, 1.15, 0.4),    // ribcage
    ellipsoid(0, 0.58, -2.95, 2.0, 0.52, 0.55, 0.4),     // shoulders
    cone([0, 0.62, -3.3], [0, 0.66, -3.95], 0.42, 0.38, 0.25), // neck
    ellipsoid(0, 0.9, -4.75, 0.72, 0.82, 0.95, 0.2),     // head
    ellipsoid(0, 1.68, -4.98, 0.1, 0.13, 0.2, 0.1),      // nose
    ellipsoid(0, 1.42, -4.18, 0.3, 0.24, 0.27, 0.15),    // chin / jaw
    ellipsoid(0, 1.45, -5.25, 0.5, 0.22, 0.3, 0.2),      // brow
  ];
  for (const s of [1, -1]) {
    P.push(
      ellipsoid(0.6 * s, 1.3, -2.05, 0.5, 0.33, 0.46, 0.25),             // breast
      cone([1.9 * s, 0.6, -2.85], [2.05 * s, 0.45, -0.7], 0.42, 0.33, 0.3), // upper arm
      cone([2.05 * s, 0.45, -0.7], [1.95 * s, 0.36, 1.35], 0.32, 0.23, 0.15), // forearm
      ellipsoid(1.88 * s, 0.24, 1.95, 0.24, 0.12, 0.42, 0.12),           // hand
      cone([0.82 * s, 0.62, 2.3], [0.72 * s, 0.55, 6.3], 0.9, 0.55, 0.45),  // thigh
      sphere(0.72 * s, 0.6, 6.35, 0.5, 0.15),                            // knee
      cone([0.72 * s, 0.55, 6.3], [0.66 * s, 0.4, 10.2], 0.52, 0.28, 0.2),  // calf
      cone([0.66 * s, 0.42, 10.4], [0.7 * s, 1.25, 10.75], 0.3, 0.2, 0.15), // foot
      sphere(0.66 * s, 0.3, 10.45, 0.28, 0.1),                           // heel
      ellipsoid(0.72 * s, 0.95, -4.75, 0.08, 0.2, 0.15, 0.08),           // ear
    );
  }
  return makeSDF(P);
}

export function buildMother() {
  const sdf = motherSDF();
  const geo = polygonise(sdf, [0, 1.0, 2.6], [2.6, 1.15, 8.6], 150, 420000);
  // height table over the abdomen for fast, exact probe placement on this same surface
  const x0 = -1.8, z0 = -2.3, step = 0.025, nx = 145, nz = 197;
  const H = new Float32Array(nx * nz);
  for (let j = 0; j < nz; j++) {
    const z = z0 + j * step;
    const list = sdf.cull(z, undefined, 0.05);
    for (let i = 0; i < nx; i++) {
      const x = x0 + i * step;
      let lo = 0, hi = 2.6;
      if (sdf.evalList(list, x, 0.02, z) > 0) { H[j * nx + i] = 0; continue; }
      for (let it = 0; it < 22; it++) {
        const mid = (lo + hi) / 2;
        if (sdf.evalList(list, x, mid, z) < 0) lo = mid; else hi = mid;
      }
      H[j * nx + i] = lo;
    }
  }
  const table = { H, x0, z0, step, nx, nz };
  return { geometry: geo, table };
}

export function makeSurface(table) {
  const { H, x0, z0, step, nx, nz } = table;
  function y(x, z) {
    const fx = Math.min(nx - 1.001, Math.max(0, (x - x0) / step));
    const fz = Math.min(nz - 1.001, Math.max(0, (z - z0) / step));
    const i = Math.floor(fx), j = Math.floor(fz), a = fx - i, b = fz - j;
    const h00 = H[j * nx + i], h10 = H[j * nx + i + 1], h01 = H[(j + 1) * nx + i], h11 = H[(j + 1) * nx + i + 1];
    return (h00 * (1 - a) + h10 * a) * (1 - b) + (h01 * (1 - a) + h11 * a) * b;
  }
  function normal(x, z) {
    const e = step * 1.5;
    return new THREE.Vector3(-(y(x + e, z) - y(x - e, z)) / (2 * e), 1, -(y(x, z + e) - y(x, z - e)) / (2 * e)).normalize();
  }
  return { y, normal };
}

// ------------------------------------------------------------------------------ the fetus
// Canonical pose, fetal units. +x towards the crown, +y dorsal (the back), z lateral.
// Flexed mid-trimester attitude: large head, chin tucked, C-curved spine, limbs folded.
export const FETAL_LANDMARKS = {
  HC: new THREE.Vector3(0.38, 0.02, 0),         // centre of the head (transthalamic plane)
  AC: new THREE.Vector3(-0.14, -0.03, 0),       // centre of the abdomen (stomach / umbilical vein)
  FL: new THREE.Vector3(-0.2, -0.18, 0.115),    // mid-shaft of the near femur (thigh flexed onto the belly)
  navel: new THREE.Vector3(-0.15, -0.19, 0),
};

function fetusSDF() {
  const C = [0.38, 0.02, 0];
  const at = (dx, dy, dz) => [C[0] + dx, C[1] + dy, C[2] + dz];
  const add = [
    ellipsoid(...at(0, 0, 0), 0.24, 0.22, 0.2, 0.05),            // cranium
    ellipsoid(...at(-0.05, 0.05, 0), 0.2, 0.18, 0.18, 0.06),     // occiput
    ellipsoid(...at(0.06, -0.12, 0), 0.14, 0.12, 0.14, 0.06),    // midface / cheeks
    ellipsoid(...at(0.14, -0.14, 0), 0.07, 0.07, 0.125, 0.04),   // forehead / brow ridge
    ellipsoid(...at(0.075, -0.205, 0), 0.05, 0.025, 0.02, 0.03), // nose bridge
    ellipsoid(...at(0.035, -0.235, 0), 0.03, 0.03, 0.028, 0.025),// nose tip
    ellipsoid(...at(-0.025, -0.215, 0), 0.026, 0.024, 0.048, 0.015), // upper lip
    ellipsoid(...at(-0.055, -0.205, 0), 0.022, 0.02, 0.04, 0.015),   // lower lip
    ellipsoid(...at(-0.1, -0.17, 0), 0.06, 0.06, 0.08, 0.04),    // chin / jaw
    ellipsoid(...at(-0.08, -0.12, 0.1), 0.08, 0.07, 0.06, 0.05), // jaw angle
    ellipsoid(...at(-0.08, -0.12, -0.1), 0.08, 0.07, 0.06, 0.05),
    cone(at(-0.16, -0.06, 0), [0.12, -0.01, 0], 0.095, 0.11, 0.06), // neck
    cone([0.11, 0.0, 0], [-0.07, 0.0, 0], 0.15, 0.165, 0.08),     // thorax
    ellipsoid(-0.14, -0.03, 0, 0.15, 0.16, 0.155, 0.08),          // abdomen
    ellipsoid(-0.28, 0.03, 0, 0.11, 0.13, 0.135, 0.07),           // pelvis / buttocks
    sphere(-0.15, -0.185, 0, 0.028, 0.03),                        // cord insertion
  ];
  for (const s of [1, -1]) {
    add.push(
      ellipsoid(...at(-0.03, -0.01, 0.2 * s), 0.045, 0.055, 0.02, 0.02), // ear
      sphere(0.1, 0.0, 0.14 * s, 0.06, 0.05),                              // shoulder
    );
  }
  // spine: vertebral ridge along the C-curved back
  for (let i = 0; i < 14; i++) {
    const t = i / 13;
    const x = 0.13 - t * 0.44, y = 0.135 + 0.045 * Math.sin(Math.PI * t);
    add.push(sphere(x, y, 0, 0.032, 0.035));
  }
  // arms: right hand at the mouth (thumb-sucking), left folded on the chest
  const arm = (s, sh, el, wr, hand) => {
    add.push(cone(sh, el, 0.05, 0.04, 0.04), cone(el, wr, 0.038, 0.03, 0.025),
      ellipsoid(...hand, 0.048, 0.024, 0.038, 0.025),
      cone(hand, [hand[0] + 0.05, hand[1] - 0.005, hand[2] - 0.03 * s], 0.016, 0.012, 0.012)); // thumb
  };
  arm(1, [0.1, -0.02, 0.15], [-0.03, -0.15, 0.17], [0.22, -0.22, 0.08], [0.28, -0.235, 0.04]);
  arm(-1, [0.1, -0.02, -0.15], [-0.05, -0.13, -0.18], [0.1, -0.2, -0.1], [0.16, -0.22, -0.06]);
  // legs: flexed at hip and knee, feet crossed in front of the pelvis
  const leg = (s, hip, knee, ankle, toe) => {
    add.push(cone(hip, knee, 0.075, 0.05, 0.05), sphere(...knee, 0.052, 0.03),
      cone(knee, ankle, 0.05, 0.032, 0.03), sphere(...ankle, 0.036, 0.02),
      cone(ankle, toe, 0.034, 0.03, 0.02));
  };
  leg(1, [-0.28, -0.06, 0.09], [-0.12, -0.3, 0.14], [-0.33, -0.37, 0.07], [-0.28, -0.43, 0.03]);
  leg(-1, [-0.28, -0.06, -0.09], [-0.1, -0.31, -0.14], [-0.31, -0.39, -0.06], [-0.26, -0.45, 0.02]);
  const sub = [
    sphere(...at(0.1, -0.168, 0.068), 0.028, 0.03),     // shallow orbits
    sphere(...at(0.1, -0.168, -0.068), 0.028, 0.03),
    cone(at(-0.025, -0.235, -0.03), at(-0.025, -0.235, 0.03), 0.006, 0.006, 0.01), // mouth line
  ];
  // closed eyelids fill the orbits
  const lids = [ellipsoid(...at(0.098, -0.185, 0.068), 0.028, 0.02, 0.026, 0.02),
    ellipsoid(...at(0.098, -0.185, -0.068), 0.028, 0.02, 0.026, 0.02)];
  return makeSDF(add, sub, lids);
}

export function buildFetus() {
  return polygonise(fetusSDF(), [0.1, -0.08, 0], [0.58, 0.45, 0.3], 112, 160000);
}

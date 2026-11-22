// 3D scene: translucent maternal abdomen, uterus, fetus, handheld probe with acoustic beam.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

import { BELLY, SCAN, buildMother, buildFetus, makeSurface, FETAL_LANDMARKS } from "./anatomy.js";

// ---- body geometry (one unit = 10 cm) -------------------------------------------------
const FETAL_DEPTH = 0.95;                               // y of the fetal centre plane
const MOTHER = buildMother();                           // smooth SDF patient + height table
const SURF = makeSurface(MOTHER.table);
const FETUS_GEO = buildFetus();                         // canonical fetus, transformed per episode

export function surfaceY(x, z) { return SURF.y(x, z); }
function surfaceNormal(x, z) { return SURF.normal(x, z); }
// env (u, v) in [0,1]^2 -> world x/z. u grows to maternal right, v grows cranially (-z).
export function uvToXZ(u, v) {
  return [(u - 0.5) * 2 * BELLY.x * SCAN, BELLY.zc - (v - 0.5) * 2 * BELLY.z * SCAN];
}
function uvDirToWorld(du, dv) {
  return new THREE.Vector3(du * BELLY.x, 0, -dv * BELLY.z).normalize();
}

// Translucent tissue: physical material with a fresnel-driven alpha (medical-plate look).
function tissueMaterial({ color, minA, maxA, rim, sheen = 0.3, contours = false }) {
  const m = new THREE.MeshPhysicalMaterial({
    color, roughness: 0.6, metalness: 0, sheen, sheenColor: new THREE.Color(rim), sheenRoughness: 0.6,
    transparent: true, depthWrite: false, side: THREE.FrontSide,
  });
  m.userData.uniforms = {
    uMinA: { value: minA }, uMaxA: { value: maxA }, uRim: { value: new THREE.Color(rim) }, uFade: { value: 1 },
    uScan: { value: new THREE.Vector3(BELLY.x * SCAN, BELLY.z * SCAN, BELLY.zc) }, uContours: { value: contours ? 1 : 0 },
  };
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, m.userData.uniforms);
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vWP;")
      .replace("#include <project_vertex>", "#include <project_vertex>\nvWP = (modelMatrix * vec4(transformed, 1.0)).xyz;");
    sh.fragmentShader = sh.fragmentShader
      .replace("#include <common>", "#include <common>\nuniform float uMinA; uniform float uMaxA; uniform vec3 uRim; uniform float uFade; uniform vec3 uScan; uniform float uContours; varying vec3 vWP;")
      .replace("#include <opaque_fragment>", `#include <opaque_fragment>
        float fres = pow(1.0 - clamp(abs(dot(normalize(normal), normalize(vViewPosition))), 0.0, 1.0), 2.2);
        gl_FragColor.rgb += uRim * fres * 0.22;
        float alpha = mix(uMinA, uMaxA, fres);
        if (uContours > 0.5) {
          // faint topographic contours give the dome its form
          float h = vWP.y / 0.13;
          float eb = length(vec2(vWP.x / uScan.x, (vWP.z - uScan.z) / uScan.y));
          float cl = (1.0 - smoothstep(0.0, 1.2, abs(fract(h - 0.5) - 0.5) / fwidth(h))) * (1.0 - smoothstep(1.05, 1.5, eb));
          gl_FragColor.rgb = mix(gl_FragColor.rgb, vec3(1.0, 0.9, 0.8), cl * 0.12);
          alpha += cl * 0.06;
          // scannable region outline (the agent's action space boundary)
          float e = length(vec2(vWP.x / uScan.x, (vWP.z - uScan.z) / uScan.y));
          float ring = 1.0 - smoothstep(0.0, 1.5, abs(e - 1.0) / fwidth(e));
          float dash = step(0.5, fract(atan(vWP.x / uScan.x, (vWP.z - uScan.z) / uScan.y) * 24.0 / 6.2832));
          gl_FragColor.rgb = mix(gl_FragColor.rgb, vec3(0.91, 0.64, 0.24), ring * dash * 0.8);
          alpha += ring * dash * 0.45;
        }
        gl_FragColor.a = alpha * uFade;`);
  };
  return m;
}

// Fan-shaped acoustic beam with an animated scan line.
function beamMaterial() {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, depthTest: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
    uniforms: { uTime: { value: 0 }, uQ: { value: 0 }, uColor: { value: new THREE.Color("#e8a33d") } },
    vertexShader: `varying vec2 vP; void main(){ vP = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
    fragmentShader: `
      varying vec2 vP; uniform float uTime; uniform float uQ; uniform vec3 uColor;
      float ss(float a, float b, float x){ float t = clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
      void main(){
        float r = length(vP) / 1.4;
        float ang = atan(vP.x, -vP.y) / 0.62;               // -1..1 across the sector
        float edge = ss(1.0, 0.92, abs(ang));
        float fall = (1.0 - r) * (1.0 - r);
        float scan = exp(-pow((ang - sin(uTime * 2.2)) * 9.0, 2.0));
        float rings = 0.5 + 0.5 * sin(r * 70.0 - uTime * 6.0);
        float a = edge * fall * (0.22 + 0.18 * uQ + 0.35 * scan + 0.05 * rings);
        vec3 c = mix(vec3(0.95, 0.92, 0.86), uColor, 0.35 + 0.5 * uQ);
        gl_FragColor = vec4(c * a, a);
      }`,
  });
}

function sectorGeometry(depth = 1.4, half = 0.62, n = 48) {
  const pos = [0, 0, 0], idx = [];
  for (let i = 0; i <= n; i++) {
    const a = -half + (2 * half * i) / n;
    pos.push(depth * Math.sin(a), -depth * Math.cos(a), 0);
    if (i > 0) idx.push(0, i, i + 1);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  return g;
}

function buildProbe() {
  const probe = new THREE.Group();
  const shell = new THREE.MeshPhysicalMaterial({ color: "#e4dfd6", roughness: 0.32, clearcoat: 0.6, clearcoatRoughness: 0.25 });
  const grip = new THREE.MeshPhysicalMaterial({ color: "#2a2d2f", roughness: 0.7 });
  const lens = new THREE.MeshPhysicalMaterial({ color: "#3a3f42", roughness: 0.25, clearcoat: 1 });

  // Handle: lathe profile, then flattened so the grip is ergonomic (wider along the array axis).
  const prof = [];
  const P = [[0.0, 0.0], [0.17, 0.0], [0.19, 0.05], [0.18, 0.14], [0.12, 0.30], [0.105, 0.48], [0.11, 0.62], [0.09, 0.72], [0.05, 0.78], [0.0, 0.79]];
  P.forEach(([r, y]) => prof.push(new THREE.Vector2(r, y)));
  const handle = new THREE.Mesh(new THREE.LatheGeometry(prof, 64), shell);
  handle.scale.set(1, 1, 0.55);
  handle.position.y = 0.03;
  handle.castShadow = true;
  probe.add(handle);

  const band = new THREE.Mesh(new THREE.CylinderGeometry(0.112, 0.118, 0.16, 64, 1, true), grip);
  band.scale.set(1, 1, 0.56);
  band.position.y = 0.42;
  probe.add(band);

  // Curvilinear transducer face.
  const face = new THREE.Mesh(new THREE.CapsuleGeometry(0.045, 0.28, 8, 24), lens);
  face.rotation.z = Math.PI / 2;
  face.scale.set(1, 1, 1.3);
  face.position.y = 0.03;
  probe.add(face);

  // Orientation marker (real probes have a notch / LED on one side).
  const mark = new THREE.Mesh(new THREE.SphereGeometry(0.016, 16, 16), new THREE.MeshBasicMaterial({ color: "#e8a33d" }));
  mark.position.set(0.175, 0.1, 0);
  probe.add(mark);

  // Strain relief at the top.
  const relief = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.05, 0.14, 24), grip);
  relief.position.y = 0.86;
  probe.add(relief);

  // Cable: leaves the strain relief and drapes away from the patient.
  const cablePts = [[0, 0.92, 0], [0, 1.08, 0.03], [0.04, 1.18, 0.25], [0.12, 1.05, 0.6], [0.25, 0.6, 0.95], [0.4, -0.2, 1.2]]
    .map(([x, y, z]) => new THREE.Vector3(x, y, z));
  const cable = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(cablePts), 80, 0.022, 12, false),
    new THREE.MeshPhysicalMaterial({ color: "#3a3d40", roughness: 0.55, clearcoat: 0.3 }));
  cable.castShadow = true;
  probe.add(cable);

  const beam = new THREE.Mesh(sectorGeometry(), beamMaterial());
  beam.renderOrder = 10;
  probe.add(beam);
  return { probe, beam, mark };
}

function makeLabel(text) {
  const c = document.createElement("canvas");
  c.width = 128; c.height = 64;
  const g = c.getContext("2d");
  g.font = "500 40px 'Plex Mono', monospace";
  g.fillStyle = "#e8a33d";
  g.textBaseline = "middle";
  g.fillText(text, 8, 34);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true, toneMapped: false }));
  sp.scale.set(0.32, 0.16, 1);
  sp.renderOrder = 12;
  sp.userData.canvas = c;
  return sp;
}


export class Scene3D {
  constructor(canvas) {
    this.canvas = canvas;
    const r = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" }));
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.05;
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.setClearColor(0x000000, 0);

    const s = (this.scene = new THREE.Scene());
    const pmrem = new THREE.PMREMGenerator(r);
    s.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    s.environmentIntensity = 0.35;

    this.camera = new THREE.PerspectiveCamera(30, 1, 0.05, 100);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.rotateSpeed = 0.5;
    this.controls.zoomSpeed = 0.6;
    this.controls.enablePan = false;
    this.controls.minDistance = 2.5;
    this.controls.maxDistance = 36;
    this.controls.maxPolarAngle = Math.PI * 0.47;

    this.views = {
      1: { name: "Overview", pos: new THREE.Vector3(5.2, 4.4, 5.2), look: new THREE.Vector3(0, 1.0, 0.6), follow: 0.15 },
      2: { name: "Probe close-up", pos: null, look: new THREE.Vector3(0, 1.0, BELLY.zc), follow: 1.0 },
      3: { name: "Full patient", pos: new THREE.Vector3(25, 15, 3.0), look: new THREE.Vector3(0, 0.2, 2.9), follow: 0.0 },
    };
    this.view = 1;
    this.camGoal = this.views[1].pos.clone();
    this.camera.position.copy(this.camGoal);
    this.lookGoal = new THREE.Vector3(0, 1.0, 0.1);
    this.controls.target.copy(this.lookGoal);
    this.userOrbiting = false;
    this.controls.addEventListener("start", () => (this.userOrbiting = true));

    this._lights();
    this._table();
    this._body();
    this.fetus = new THREE.Group();
    this.uterusGroup.add(this.fetus);
    const { probe, beam } = buildProbe();
    this.probe = probe;
    this.beam = beam;
    s.add(probe);
    this._trail();
    this._markers();

    this.pose = { u: 0.5, v: 0.5, theta: 0, phi: 0 };
    this.poseGoal = { ...this.pose };
    this.quality = 0;
    this.gt = false;
    this.clock = new THREE.Clock();
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  _lights() {
    const s = this.scene;
    const key = new THREE.DirectionalLight("#fff1e0", 2.4);
    key.position.set(3.5, 7, 3);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.radius = 6;
    key.shadow.bias = -0.0004;
    Object.assign(key.shadow.camera, { left: -5, right: 5, top: 5, bottom: -5, near: 1, far: 20 });
    s.add(key);
    const fill = new THREE.HemisphereLight("#cfd6dc", "#1a1512", 0.55);
    s.add(fill);
    const rim = new THREE.DirectionalLight("#ffd2a8", 1.6);
    rim.position.set(-4, 3, -5);
    s.add(rim);
  }

  _table() {
    const mat = new THREE.MeshStandardMaterial({ color: "#15191c", roughness: 0.85 });
    const couch = new THREE.Mesh(new THREE.BoxGeometry(5.6, 0.22, 18.4), mat);
    couch.position.set(0, -0.11, 2.6);
    couch.receiveShadow = true;
    this.scene.add(couch);
    const base = new THREE.Mesh(new THREE.BoxGeometry(3.2, 3.4, 12), new THREE.MeshStandardMaterial({ color: "#0f1214", roughness: 0.9 }));
    base.position.set(0, -1.92, 2.6);
    this.scene.add(base);
    const sheet = new THREE.Mesh(new THREE.PlaneGeometry(5.0, 17.6), new THREE.MeshStandardMaterial({ color: "#1f262a", roughness: 0.95 }));
    sheet.rotation.x = -Math.PI / 2;
    sheet.position.set(0, 0.002, 2.6);
    sheet.receiveShadow = true;
    this.scene.add(sheet);
    // pillow under the head
    const pillow = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), new THREE.MeshStandardMaterial({ color: "#2b3236", roughness: 0.9 }));
    pillow.scale.set(1.5, 0.22, 0.9);
    pillow.position.set(0, 0.12, -4.9);
    pillow.receiveShadow = true;
    this.scene.add(pillow);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.ShadowMaterial({ opacity: 0.35 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -3.6;
    floor.receiveShadow = true;
    this.scene.add(floor);
  }

  _body() {
    this.skinMat = tissueMaterial({ color: "#c89478", minA: 0.24, maxA: 0.75, rim: "#ffcfae", sheen: 0.5, contours: true });
    const skin = new THREE.Mesh(MOTHER.geometry, this.skinMat);
    skin.renderOrder = 5;
    this.scene.add(skin);

    this.uterusGroup = new THREE.Group();
    this.scene.add(this.uterusGroup);
    this.uterusMat = tissueMaterial({ color: "#a8605a", minA: 0.04, maxA: 0.34, rim: "#e59a8a", sheen: 0.2 });
    const ut = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 64), this.uterusMat);
    ut.scale.set(1.12, 0.8, 1.38);
    ut.position.set(0, FETAL_DEPTH, BELLY.zc);
    this.uterus = ut;
    ut.renderOrder = 4;
    this.uterusGroup.add(ut);

    const plac = new THREE.Mesh(
      new THREE.SphereGeometry(1, 48, 32),
      new THREE.MeshPhysicalMaterial({ color: "#4a2828", roughness: 0.85, sheen: 0.3, sheenColor: new THREE.Color("#8a5048") })
    );
    plac.scale.set(0.56, 0.13, 0.5);
    plac.position.set(-0.25, FETAL_DEPTH - 0.62, BELLY.zc - 0.35);
    plac.rotation.z = -0.12;
    this.placenta = plac;
    this.uterusGroup.add(plac);
  }

  _trail() {
    this.trailN = 360;
    const g = new THREE.CircleGeometry(0.018, 16);
    const m = new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, vertexColors: false, toneMapped: false });
    this.trail = new THREE.InstancedMesh(g, m, this.trailN);
    this.trail.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.trail.count = 0;
    this.trail.renderOrder = 6;
    this.trailPts = [];
    this.scene.add(this.trail);
  }

  _markers() {
    this.markers = new THREE.Group();
    this.markers.visible = false;
    this.scene.add(this.markers);
    this.markerObjs = {};
    for (const k of ["HC", "AC", "FL"]) {
      const g = new THREE.Group();
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.1, 0.118, 48), new THREE.MeshBasicMaterial({ color: "#e8a33d", transparent: true, opacity: 0.9, depthWrite: false, toneMapped: false }));
      const bar = new THREE.Mesh(new THREE.PlaneGeometry(0.42, 0.014), new THREE.MeshBasicMaterial({ color: "#e8a33d", transparent: true, opacity: 0.7, depthWrite: false, toneMapped: false }));
      ring.renderOrder = bar.renderOrder = 7;
      const label = makeLabel(k);
      label.position.set(0.16, 0.16, 0.02);
      g.add(ring, bar, label);
      g.userData = { ring, bar, label };
      this.markers.add(g);
      this.markerObjs[k] = g;
    }
    // vertical drop line from each marker to the anatomy
    this.dropLines = {};
    for (const k of ["HC", "AC", "FL"]) {
      const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
      const l = new THREE.Line(geo, new THREE.LineDashedMaterial({ color: "#e8a33d", dashSize: 0.04, gapSize: 0.04, transparent: true, opacity: 0.6 }));
      this.markers.add(l);
      this.dropLines[k] = l;
    }
  }

  // ------------------------------------------------------------------------- fetus
  setFetus(fetus, targets) {
    this.fetus.clear();
    this.targets = targets;
    const at = (k) => {
      const [x, z] = uvToXZ(targets[k][0], targets[k][1]);
      return new THREE.Vector3(x, FETAL_DEPTH, z);
    };
    const H = at("HC"), A = at("AC");
    const L = FETAL_LANDMARKS;
    // scale so the canonical head-to-abdomen distance matches the hidden targets
    const lhA = new THREE.Vector2(L.HC.x - L.AC.x, L.HC.z - L.AC.z).length();
    const s = H.distanceTo(A) / lhA;
    const d = H.clone().sub(A).normalize();                     // world direction to the crown
    const up = new THREE.Vector3(0, 1, 0);
    // semi-lateral lie: back rolled ~35 deg down to one side, face / folded limbs towards the probe.
    // femur_side mirrors the pose across the vertical plane through the fetal axis.
    const h = new THREE.Vector3().crossVectors(up, d).normalize();
    const sd = fetus.femur_side, ca = Math.cos(0.61), sa = Math.sin(0.61);
    const Y = h.clone().multiplyScalar(sd * ca).addScaledVector(up, -sa);
    const Z = up.clone().multiplyScalar(ca).addScaledVector(h, sd * sa);
    const basis = new THREE.Matrix4().makeBasis(d, Y, Z);
    const m = new THREE.Matrix4().makeTranslation(A.x, A.y, A.z)
      .multiply(basis).multiply(new THREE.Matrix4().makeScale(s, s, s))
      .multiply(new THREE.Matrix4().makeTranslation(-L.AC.x, -L.AC.y, -L.AC.z));

    const skinM = new THREE.MeshPhysicalMaterial({
      color: "#b9746a", roughness: 0.5, sheen: 0.45, sheenColor: new THREE.Color("#f3b49c"), sheenRoughness: 0.5,
      emissive: new THREE.Color("#4a160c"), emissiveIntensity: 0.1,
      clearcoat: 0.25, clearcoatRoughness: 0.35,   // amniotic sheen
    });
    const body = new THREE.Mesh(FETUS_GEO, skinM);
    body.matrixAutoUpdate = false;
    body.matrix.copy(m);
    body.castShadow = true;
    this.fetus.add(body);

    // landmark positions in world space (for ground-truth drop lines and the cord)
    const W = (v) => v.clone().applyMatrix4(m);
    this.anatomy = { H: W(L.HC), A: W(L.AC), F: W(L.FL) };

    // coiled umbilical cord from the navel to the placenta
    const navel = W(L.navel);
    const pl = this.placenta.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0, 0.08, 0));
    const pts = [];
    const off = new THREE.Vector3().crossVectors(d, up).multiplyScalar(0.3);
    const base = new THREE.CubicBezierCurve3(navel, navel.clone().lerp(pl, 0.3).add(off).add(new THREE.Vector3(0, 0.1, 0)),
      navel.clone().lerp(pl, 0.7).sub(off), pl);
    for (let i = 0; i <= 160; i++) {
      const t = i / 160, p = base.getPoint(t), tan = base.getTangent(t);
      const n1 = new THREE.Vector3().crossVectors(tan, up).normalize();
      const n2 = new THREE.Vector3().crossVectors(tan, n1).normalize();
      const a = t * Math.PI * 2 * 7, r = 0.012;
      pts.push(p.add(n1.multiplyScalar(Math.cos(a) * r)).add(n2.multiplyScalar(Math.sin(a) * r)));
    }
    const cordM = new THREE.MeshPhysicalMaterial({ color: "#c9ced8", roughness: 0.35, clearcoat: 0.6, sheen: 0.5, sheenColor: new THREE.Color("#eef2f7") });
    const cord = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 400, 0.022, 10, false), cordM);
    cord.castShadow = true;
    this.fetus.add(cord);

    // the uterus follows the fetus a little so the fetus stays inside it
    this.uterus.position.set(A.x * 0.4, FETAL_DEPTH, BELLY.zc + (A.z - BELLY.zc) * 0.4);
    this._placeMarkers(targets);
  }

  _placeMarkers(targets) {
    for (const k of ["HC", "AC", "FL"]) {
      const [u, v, th] = targets[k];
      const { pos, n, axis } = this._frame(u, v, th);
      const g = this.markerObjs[k];
      g.position.copy(pos).addScaledVector(n, 0.012);
      const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), n);
      g.quaternion.copy(q);
      const localX = new THREE.Vector3(1, 0, 0).applyQuaternion(q);
      const ang = Math.atan2(localX.clone().cross(axis).dot(n), localX.dot(axis));
      g.userData.bar.rotation.z = ang;
      const a = k === "HC" ? this.anatomy.H : k === "AC" ? this.anatomy.A : this.anatomy.F;
      const l = this.dropLines[k];
      l.geometry.setFromPoints([pos, a]);
      l.computeLineDistances();
    }
  }

  // Local probe frame at (u, v) with rotation theta: position, normal, array axis.
  _frame(u, v, theta) {
    const [x, z] = uvToXZ(u, v);
    const pos = new THREE.Vector3(x, surfaceY(x, z), z);
    const n = surfaceNormal(x, z);
    const tu = uvDirToWorld(1, 0), tv = uvDirToWorld(0, 1);
    const dir = tu.multiplyScalar(Math.cos(theta)).add(tv.multiplyScalar(Math.sin(theta)));
    const axis = dir.sub(n.clone().multiplyScalar(dir.dot(n))).normalize();
    return { pos, n, axis };
  }

  _applyProbe(p) {
    const { pos, n, axis } = this._frame(p.u, p.v, p.theta);
    const z = new THREE.Vector3().crossVectors(axis, n).normalize();
    const m = new THREE.Matrix4().makeBasis(axis, n, z);
    const q = new THREE.Quaternion().setFromRotationMatrix(m);
    q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), p.phi)); // fan about the array axis
    this.probe.position.copy(pos);
    this.probe.quaternion.copy(q);
    return pos;
  }

  // ------------------------------------------------------------------------- updates
  setState(st, instant = false) {
    const p = st.probe;
    // unwrap theta so interpolation takes the short way round
    let th = p.theta;
    while (th - this.poseGoal.theta > Math.PI) th -= 2 * Math.PI;
    while (th - this.poseGoal.theta < -Math.PI) th += 2 * Math.PI;
    this.poseGoal = { u: p.u, v: p.v, theta: th, phi: p.phi };
    if (instant) this.pose = { ...this.poseGoal };
    this.quality = st.quality;
    if (st.targets && this.targets) {
      const moved = ["HC", "AC", "FL"].some((k) => Math.abs(st.targets[k][0] - this.targets[k][0]) > 1e-6);
      if (moved) this.setFetus(st.fetus, st.targets);
    }
    this.captured = st.captured;
    for (const k of ["HC", "AC", "FL"]) {
      const g = this.markerObjs[k];
      const col = st.captured[k] ? "#6faf8f" : "#e8a33d";
      g.userData.ring.material.color.set(col);
      g.userData.bar.material.color.set(col);
      g.userData.label.material.color.set(st.captured[k] ? "#9fd1b6" : "#ffffff");
      this.dropLines[k].material.color.set(col);
    }
  }

  addTrailPoint(st) {
    const [x, z] = uvToXZ(st.probe.u, st.probe.v);
    this.trailPts.push({ x, z, q: st.quality, age: 0 });
    if (this.trailPts.length > this.trailN) this.trailPts.shift();
  }

  resetEpisode(st) {
    this.trailPts = [];
    this.setFetus(st.fetus, st.targets);
    this.poseGoal = { ...st.probe };
    this.setState(st, true);
  }

  setGroundTruth(on) {
    this.gt = on;
    this.markers.visible = on;
  }

  setView(k) {
    if (!this.views[k]) return;
    this.view = k;
    this.userOrbiting = false;
    return this.views[k].name;
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render() {
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;
    const k = 1 - Math.exp(-dt * 10);
    for (const key of ["u", "v", "theta", "phi"]) this.pose[key] += (this.poseGoal[key] - this.pose[key]) * k;
    const pos = this._applyProbe(this.pose);
    this.beam.material.uniforms.uTime.value = t;
    this.beam.material.uniforms.uQ.value += (this.quality - this.beam.material.uniforms.uQ.value) * k;

    // skin fades further in ground-truth mode so the anatomy reads clearly
    const fadeGoal = this.gt ? 0.55 : 1.0;
    const u = this.skinMat.userData.uniforms.uFade;
    u.value += (fadeGoal - u.value) * k;

    // trail: dots on the skin, coloured by quality, fading with age
    const m = new THREE.Matrix4(), col = new THREE.Color(), dim = new THREE.Color("#8c877f"), amb = new THREE.Color("#e8a33d");
    const n = this.trailPts.length;
    this.trail.count = n;
    for (let i = 0; i < n; i++) {
      const p = this.trailPts[i];
      const y = surfaceY(p.x, p.z);
      const nn = surfaceNormal(p.x, p.z);
      const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), nn);
      const age = (n - 1 - i) / this.trailN;
      const s = 0.55 + 0.9 * p.q;
      m.compose(new THREE.Vector3(p.x, y, p.z).addScaledVector(nn, 0.008), q, new THREE.Vector3(s, s, s));
      this.trail.setMatrixAt(i, m);
      col.copy(dim).lerp(amb, Math.min(1, p.q * 1.3)).multiplyScalar(Math.max(0.08, 1 - age * 1.6));
      this.trail.setColorAt(i, col);
    }
    this.trail.instanceMatrix.needsUpdate = true;
    if (this.trail.instanceColor) this.trail.instanceColor.needsUpdate = true;
    this.trail.material.opacity = 0.9;

    // camera: gentle follow unless the user is orbiting
    const view = this.views[this.view];
    const look = view.look.clone().lerp(pos, view.follow);
    this.lookGoal.lerp(look, 1 - Math.exp(-dt * 2.5));
    this.controls.target.lerp(this.lookGoal, 1 - Math.exp(-dt * 3));
    if (!this.userOrbiting) {
      const goal = view.pos ? view.pos : pos.clone().add(new THREE.Vector3(2.0, 2.2, 2.6));
      this.camera.position.lerp(goal, 1 - Math.exp(-dt * 1.8));
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

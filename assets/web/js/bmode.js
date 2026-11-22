// Simulated B-mode sector image (fragment shader). Anatomy fades in as alignment rises.
import * as THREE from "three";

const FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform vec2 uRes;
uniform float uTime;
uniform vec3 uA;        // alignment per plane (HC, AC, FL)
uniform vec3 uVis;      // coarse visibility per plane
uniform vec3 uLat;      // lateral offset of each anatomy in the image (-1..1)
uniform float uEdge;    // acoustic shadowing near the abdomen edge
uniform float uSeed;
uniform float uFrozen;

#define HALF 0.62
#define R0 0.10
#define R1 1.02

float ss(float a, float b, float x){ float t = clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
float hash(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  float a = hash(i), b = hash(i + vec2(1,0)), c = hash(i + vec2(0,1)), d = hash(i + vec2(1,1));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++){ s += a * vnoise(p); p *= 2.03; a *= 0.5; } return s; }
float line(float d, float w){ return exp(-(d * d) / (w * w)); }

void main(){
  float aspect = uRes.x / uRes.y;
  // apex at the top centre; q in 'image' units where depth 0..1 ~ 0..14 cm
  vec2 q = vec2((vUv.x - 0.5) * aspect * 1.08, (1.0 - vUv.y) * 1.08 + 0.02);
  float r = length(q);
  float ang = atan(q.x, q.y);
  float inside = step(abs(ang), HALF) * step(R0, r) * step(r, R1);
  float edgeSoft = ss(HALF, HALF - 0.015, abs(ang)) * ss(R1, R1 - 0.01, r);
  float dn = clamp((r - R0) / (R1 - R0), 0.0, 1.0);   // normalised depth
  float lat = ang / HALF;                             // -1..1 across the sector

  // ---- speckle: anisotropic in polar coords, slow temporal decorrelation
  float t = floor(uTime * 24.0) * (1.0 - uFrozen);
  vec2 sp = vec2(ang * 170.0, r * 330.0);
  float sk = hash(floor(sp) + uSeed + t * 0.37);
  sk = mix(sk, hash(floor(sp * 1.7) + 7.0 + t * 0.21), 0.45);
  float speck = pow(sk, 1.6) * 1.6;

  // ---- tissue layers (wavy interfaces)
  float w = (fbm(vec2(lat * 3.0 + uSeed, 1.0)) - 0.5) * 0.05;
  float E = 0.0;
  E += 0.95 * line(dn - 0.012, 0.012);                                  // skin line
  float fat = ss(0.02, 0.03, dn) * (1.0 - ss(0.09 + w, 0.10 + w, dn));
  E += fat * (0.22 + 0.35 * line(fract(dn * 38.0 + fbm(vec2(lat * 6.0, dn * 4.0))) - 0.5, 0.06));
  E += 0.75 * line(dn - (0.11 + w), 0.012);                             // fascia
  float myo = ss(0.11 + w, 0.125 + w, dn) * (1.0 - ss(0.19 + w, 0.205 + w, dn));
  E += myo * 0.42;
  E += 0.55 * line(dn - (0.2 + w), 0.01);                               // inner uterine wall
  float fluid = ss(0.2 + w, 0.21 + w, dn) * (1.0 - ss(0.86 - w, 0.88 - w, dn));
  E += fluid * 0.035;                                                   // amniotic fluid: anechoic
  float post = ss(0.86 - w, 0.88 - w, dn);
  E += post * (0.38 + 0.2 * fbm(vec2(lat * 4.0, dn * 9.0)));            // posterior wall / maternal tissue

  // ---- fetal anatomy in Cartesian image space
  vec2 c0 = vec2(0.0, 0.56);
  float shadow = 0.0;

  // coarse fetal tissue (visible before a standard plane is reached)
  for (int k = 0; k < 3; k++){
    float vis = uVis[k] * (1.0 - ss(0.4, 0.9, uA[k]));
    vec2 c = c0 + vec2(uLat[k] * 0.36, 0.03 * float(k));
    vec2 d = (q - c) / vec2(0.26, 0.17);
    float blob = ss(1.0, 0.55, length(d) + 0.25 * (fbm(q * 9.0 + float(k)) - 0.5));
    E = mix(E, 0.24 + 0.18 * fbm(q * 22.0), blob * vis * 0.85);
  }

  // HC: ellipse with bright skull, midline falx, hypoechoic thalami
  {
    float a = uA.x; float m = ss(0.12, 0.85, a);
    vec2 c = c0 + vec2(uLat.x * 0.36, 0.0);
    vec2 d = (q - c) / vec2(0.25, 0.19);
    float rr = length(d);
    float blur = 1.0 + 3.0 * (1.0 - a);
    float inner = ss(1.0, 0.9, rr);
    E = mix(E, 0.16 + 0.12 * fbm(q * 30.0), inner * m);
    E += m * 1.25 * line(rr - 1.0, 0.045 * blur);
    E += m * 0.65 * line(d.x, 0.012 * blur) * step(rr, 0.85) * step(0.18, abs(d.y) + 0.2 * step(abs(d.x), 0.01)); // falx
    float th = ss(0.24, 0.16, length((d - vec2(0.0, 0.12)) / vec2(0.55, 0.35)));
    E = mix(E, 0.06, th * m * 0.8);
    float cavum = ss(0.1, 0.05, length((d - vec2(0.0, -0.32)) / vec2(0.8, 0.45)));
    E = mix(E, 0.03, cavum * m * 0.8);
    shadow += m * 0.35 * ss(0.98, 1.05, rr) * step(c.y, q.y) * step(abs(d.x), 0.9);
  }
  // AC: round abdomen, dark stomach bubble, umbilical vein, spine with shadow
  {
    float a = uA.y; float m = ss(0.12, 0.85, a);
    vec2 c = c0 + vec2(uLat.y * 0.36, 0.02);
    vec2 d = (q - c) / 0.21;
    float rr = length(d);
    float blur = 1.0 + 3.0 * (1.0 - a);
    E = mix(E, 0.32 + 0.22 * fbm(q * 26.0), ss(1.0, 0.94, rr) * m);
    E += m * 0.55 * line(rr - 1.0, 0.04 * blur);
    float st = ss(0.32, 0.22, length((d - vec2(-0.32, -0.05)) / vec2(1.0, 0.75)));
    E = mix(E, 0.02, st * m);
    float uv = line(length((d - vec2(0.28, -0.05)) / vec2(1.0, 1.8)) - 0.22, 0.05 * blur) * step(d.x, 0.45);
    E = mix(E, 0.05, uv * m * 0.9);
    for (int i = 0; i < 3; i++){
      vec2 sp = vec2(-0.18 + 0.18 * float(i), 0.78 - 0.05 * abs(float(i) - 1.0));
      float b = ss(0.12 * blur, 0.03, length(d - sp));
      E += m * 1.1 * b;
    }
    shadow += m * 0.6 * step(0.85, d.y) * ss(0.42, 0.25, abs(d.x)) * step(rr, 3.5);
  }
  // FL: bright linear femur with acoustic shadow
  {
    float a = uA.z; float m = ss(0.12, 0.85, a);
    vec2 c = c0 + vec2(uLat.z * 0.36, 0.04);
    vec2 d = q - c;
    float ca = cos(0.18), sa = sin(0.18);
    vec2 e = vec2(ca * d.x - sa * d.y, sa * d.x + ca * d.y);
    float blur = 1.0 + 3.0 * (1.0 - a);
    float bend = 0.03 * (e.x / 0.2) * (e.x / 0.2);
    float along = ss(0.215, 0.19, abs(e.x));
    float soft = ss(1.0, 0.7, length(e / vec2(0.3, 0.11)));
    E = mix(E, 0.28 + 0.15 * fbm(q * 24.0), soft * m * 0.8);
    E += m * 1.35 * line(e.y + bend, 0.011 * blur) * along;
    shadow += m * 0.8 * along * step(0.0, e.y + bend) * ss(0.0, 0.02, e.y + bend);
  }

  // ---- image formation
  E *= (1.0 - clamp(shadow, 0.0, 0.85));
  float atten = exp(-1.25 * dn) * (0.65 + 0.55 * dn);                  // attenuation with TGC
  float focus = 1.0 + 0.12 * line(dn - 0.55, 0.12);
  float lateralShadow = mix(1.0, 0.35, uEdge * ss(0.35, 1.0, abs(lat)) * (0.6 + 0.4 * fbm(vec2(lat * 4.0, dn))));
  float sideFall = mix(1.0, 0.75, ss(0.75, 1.0, abs(lat)));
  float I = E * speck * atten * focus * lateralShadow * sideFall;
  I = I / (I + 0.55) * 1.35;                                           // log-like compression
  I = pow(clamp(I, 0.0, 1.0), 0.92);
  vec3 col = vec3(I) * inside * edgeSoft;
  gl_FragColor = vec4(col, 1.0);
}`;

export class BMode {
  constructor(canvas, svg) {
    this.canvas = canvas;
    this.svg = svg;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.uniforms = {
      uRes: { value: new THREE.Vector2(1, 1) }, uTime: { value: 0 },
      uA: { value: new THREE.Vector3() }, uVis: { value: new THREE.Vector3() }, uLat: { value: new THREE.Vector3() },
      uEdge: { value: 0 }, uSeed: { value: Math.random() * 100 }, uFrozen: { value: 0 },
    };
    this.goal = { a: [0, 0, 0], vis: [0, 0, 0], lat: [0, 0, 0], edge: 0 };
    const mat = new THREE.ShaderMaterial({
      uniforms: this.uniforms, fragmentShader: FRAG,
      vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
    });
    this.scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));
    this.clock = new THREE.Clock();
    this.frozenUntil = 0;
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  // state from the env -> shader goals
  setState(st) {
    const P = ["HC", "AC", "FL"];
    const { u, v, theta } = st.probe;
    const ax = [Math.cos(theta), Math.sin(theta)];
    this.goal.a = P.map((k) => st.alignment[k]);
    this.goal.vis = P.map((k) => (st.visibility ? st.visibility[k] : st.alignment[k]));
    this.goal.lat = P.map((k) => {
      const [tu, tv] = st.targets[k];
      const lat = ((tu - u) * ax[0] + (tv - v) * ax[1]) / 0.12;
      return Math.max(-1.2, Math.min(1.2, lat));
    });
    this.goal.edge = Math.min(1, Math.hypot(u - 0.5, v - 0.5) / 0.5) ** 2;
  }

  freeze(ms = 450) { this.frozenUntil = performance.now() + ms; }

  snapshot() { return this.canvas.toDataURL("image/jpeg", 0.85); }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.uniforms.uRes.value.set(w, h);
    this._drawScale(w, h);
  }

  // vertical depth ruler (cm) to the right of the sector + focus marker, drawn as crisp SVG
  _drawScale(w, h) {
    const svg = this.svg;
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    const aspect = w / h, s = 1.08, R0 = 0.10, R1 = 1.02, HALF = 0.62;
    const yPx = (qy) => ((qy - 0.02) / s) * h;
    const xEdge = ((R1 * Math.sin(HALF)) / (aspect * s) + 0.5) * w;
    const x = Math.min(w - 36, xEdge + 18);
    let out = `<line x1="${x}" y1="${yPx(R0)}" x2="${x}" y2="${yPx(R1)}" stroke="rgba(232,228,220,0.14)" stroke-width="1"/>`;
    for (let cm = 0; cm <= 14; cm++) {
      const y = yPx(R0 + (R1 - R0) * (cm / 14));
      const big = cm % 5 === 0;
      out += `<line x1="${x}" y1="${y}" x2="${x + (big ? 8 : 4)}" y2="${y}" stroke="#8c877f" stroke-width="1"/>`;
      if (big) out += `<text x="${x + 12}" y="${y + 3.5}">${cm}</text>`;
    }
    out += `<text x="${x + 12}" y="${yPx(R1) + 16}">cm</text>`;
    const fy = yPx(R0 + (R1 - R0) * 0.55);
    out += `<path d="M ${x - 3} ${fy} l -7 -4.5 l 0 9 z" fill="#e8a33d"/>`;
    // orientation marker beside the apex (matches the amber notch on the probe)
    const ox = ((-R0 * Math.sin(HALF) - 0.08) / (aspect * s) + 0.5) * w;
    out += `<circle cx="${ox}" cy="${yPx(0.1)}" r="3.5" fill="none" stroke="#e8a33d" stroke-width="1.2"/>`;
    svg.innerHTML = out;
  }

  render() {
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const k = 1 - Math.exp(-dt * 9);
    const U = this.uniforms;
    U.uTime.value += dt;
    const frozen = performance.now() < this.frozenUntil;
    U.uFrozen.value = frozen ? 1 : 0;
    if (!frozen) {
      for (let i = 0; i < 3; i++) {
        U.uA.value.setComponent(i, U.uA.value.getComponent(i) + (this.goal.a[i] - U.uA.value.getComponent(i)) * k);
        U.uVis.value.setComponent(i, U.uVis.value.getComponent(i) + (this.goal.vis[i] - U.uVis.value.getComponent(i)) * k);
        U.uLat.value.setComponent(i, U.uLat.value.getComponent(i) + (this.goal.lat[i] - U.uLat.value.getComponent(i)) * k);
      }
      U.uEdge.value += (this.goal.edge - U.uEdge.value) * k;
    }
    this.renderer.render(this.scene, this.camera);
  }
}

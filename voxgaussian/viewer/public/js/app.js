// voxgaussian — live WebXR viewer (occupancy cubes + uvw atlas + gaussian disc modes)

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';

// ─── UVW bijection: voxel (u,v,w) ↔ atlas (x,y) ────────────────────────
// Same arithmetic as pipeline/uvw_atlas.py, but resolution-adaptive — the
// live pipeline runs at 128³ by default, the demo at 256³. The atlas is a
// `tilesPerRow × tilesPerCol` grid of `res × res` tiles, one tile per
// w-slice. Resolutions ≤ 256 give a 1-byte-per-axis identity bijection:
// each voxel's canonical-pass RGB byte equals its (u, v, w) byte exactly.
function tileLayout(res) {
  // Pick factorization a × b = res with a ≥ b and a as close to √res as possible.
  let a = Math.floor(Math.sqrt(res));
  while (a > 0 && res % a !== 0) a--;
  if (a === 0) a = res;
  const b = res / a;
  // Put the larger factor along the X axis so the atlas is wider than tall.
  return { tilesPerRow: Math.max(a, b), tilesPerCol: Math.min(a, b) };
}

function voxelToAtlas(u, v, w, res, tilesPerRow) {
  const tx = w % tilesPerRow;
  const ty = (w / tilesPerRow) | 0;
  return [tx * res + u, ty * res + v];
}

const config = await fetch('/__config__').then(r => r.json()).catch(() => ({ ws_port: 8765 }));
const WS_URL = `ws://${location.hostname || 'localhost'}:${config.ws_port}`;

// ─── Three.js base ──────────────────────────────────────────────────────

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.xr.enabled = true;
document.getElementById('app').appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a12);
scene.fog = new THREE.Fog(0x0a0a12, 25, 80);

const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.05, 200);
camera.position.set(8, 6, 8);
camera.lookAt(0, 1, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.listenToKeyEvents(window);   // arrow keys pan without canvas focus
controls.keyPanSpeed = 12;
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const key = new THREE.DirectionalLight(0xffffff, 0.8);
key.position.set(5, 10, 5);
scene.add(key);

const grid = new THREE.GridHelper(20, 20, 0x444466, 0x222233);
scene.add(grid);

const vrBtn = VRButton.createButton(renderer);
vrBtn.classList.add('btn');
document.getElementById('vr-btn-wrap').appendChild(vrBtn);

addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
});

// ─── Occupancy-cube renderer ────────────────────────────────────────────

const MAX_VOXELS = 300000;
const voxelGeom = new THREE.BoxGeometry(1, 1, 1);
// Unlit: voxel color comes from the inpaint RGB sample, which already has
// SDXL's lighting/shading baked in. Re-shading would double-light and
// emphasize cube facets.
const voxelMat = new THREE.MeshBasicMaterial({
  vertexColors: false,
  transparent: false,
  toneMapped: false,
});
const cubesMesh = new THREE.InstancedMesh(voxelGeom, voxelMat, MAX_VOXELS);
cubesMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_VOXELS * 3), 3);
cubesMesh.count = 0;
// Skip per-frame frustum check — Three.js can't bound an InstancedMesh by
// its instance positions (the base BoxGeometry's bounding sphere is at
// origin and tiny). Computing the bound ourselves each snapshot would cost
// more than the cull saves at typical voxel counts. Always draw all.
cubesMesh.frustumCulled = false;
scene.add(cubesMesh);

// Per-instance voxel coord (u, v, w) for the UVW canonical pass.
// Populated alongside instanceColor in renderVoxels; ignored by voxelMat.
const cubeUvwAttr = new THREE.InstancedBufferAttribute(new Float32Array(MAX_VOXELS * 3), 3);
cubesMesh.geometry.setAttribute('instanceUvw', cubeUvwAttr);

// Canonical-pass material: fragment outputs vec3(vUvw) / 255 → byte-perfect
// identity for resolutions ≤ 256. The framebuffer pixel under the cursor
// IS the voxel coord — no inverse projection, no raycast.
// (instanceMatrix is auto-prepended by Three.js for ShaderMaterial + InstancedMesh;
//  declaring it explicitly causes a duplicate-attribute compile error.)
const matCanonical = new THREE.ShaderMaterial({
  vertexShader: /* glsl */ `
    attribute vec3 instanceUvw;
    varying vec3 vUvw;
    void main() {
      vUvw = instanceUvw;
      gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    precision highp float;
    varying vec3 vUvw;
    void main() {
      gl_FragColor = vec4(vUvw / 255.0, 1.0);
    }
  `,
});

// ─── Gaussian-disc renderer ─────────────────────────────────────────────
// Each Gaussian = a billboarded plane with a radial alpha falloff.
// Built once, hidden by default; populated when Phase B emits gaussians.

const MAX_GAUSS = 1200000;   // raised for multi-sub-gauss-per-voxel
const gaussGeom = new THREE.PlaneGeometry(1, 1);
const gaussMat = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  blending: THREE.NormalBlending,
  uniforms: {
    uCentroid: { value: new THREE.Vector3(0, 1, 0) },
  },
  vertexShader: /* glsl */ `
    attribute vec3 instanceColor;
    attribute float instanceAlpha;
    attribute float instanceScale;
    attribute vec3 instanceNormal;
    uniform vec3 uCentroid;
    varying vec3 vColor;
    varying float vAlpha;
    varying vec2 vUV;
    void main() {
      vColor = instanceColor;
      vAlpha = instanceAlpha;
      vUV = uv;
      vec3 instancePos = vec3(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2]);

      // Dollhouse cutaway: cull this gauss if its normal points AWAY from
      // the chunk centroid (exterior face). Send the vertex outside the
      // clip volume so the fragment shader never runs.
      vec3 toCentroid = uCentroid - instancePos;
      float culled = dot(instanceNormal, toCentroid);
      if (culled < 0.0 && length(instanceNormal) > 0.01) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);    // outside clip space
        return;
      }

      vec3 cameraRight = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
      vec3 cameraUp    = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
      vec3 offset = (cameraRight * position.x + cameraUp * position.y) * instanceScale;
      gl_Position = projectionMatrix * viewMatrix * vec4(instancePos + offset, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    varying vec3 vColor;
    varying float vAlpha;
    varying vec2 vUV;
    void main() {
      vec2 d = vUV - 0.5;
      float r2 = dot(d, d) * 4.0;
      float falloff = exp(-r2 * 4.0);
      float a = vAlpha * falloff;
      if (a < 0.02) discard;
      gl_FragColor = vec4(vColor, a);
    }
  `,
});
const gaussMesh = new THREE.InstancedMesh(gaussGeom, gaussMat, MAX_GAUSS);
gaussMesh.frustumCulled = false;
gaussMesh.count = 0;
gaussMesh.visible = false;
const gInstColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_GAUSS * 3), 3);
const gInstAlpha = new THREE.InstancedBufferAttribute(new Float32Array(MAX_GAUSS), 1);
const gInstScale = new THREE.InstancedBufferAttribute(new Float32Array(MAX_GAUSS), 1);
const gInstNormal = new THREE.InstancedBufferAttribute(new Float32Array(MAX_GAUSS * 3), 3);
gaussMesh.geometry.setAttribute('instanceColor', gInstColor);
gaussMesh.geometry.setAttribute('instanceAlpha', gInstAlpha);
gaussMesh.geometry.setAttribute('instanceScale', gInstScale);
gaussMesh.geometry.setAttribute('instanceNormal', gInstNormal);
scene.add(gaussMesh);

// ─── State ──────────────────────────────────────────────────────────────

const _tmpMatrix = new THREE.Matrix4();
const _tmpColor = new THREE.Color();
const _tmpScale = new THREE.Vector3(1, 1, 1);
const _tmpQuat = new THREE.Quaternion();

let classColors = {};
let lastSnapshot = null;
let renderMode = 'cubes';   // 'cubes' | 'uvw' | 'gaussian'

document.querySelectorAll('#mode-toggle .btn').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('#mode-toggle .btn').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    renderMode = b.dataset.mode;
    cubesMesh.visible = renderMode === 'cubes' || renderMode === 'uvw';
    cubesMesh.material = renderMode === 'uvw' ? matCanonical : voxelMat;
    gaussMesh.visible = renderMode === 'gaussian' && gaussMesh.count > 0;
  });
});

// ─── Render functions ───────────────────────────────────────────────────

function renderVoxels(snapshot) {
  const { cells, resolution, extent, origin, class_colors, centroid } = snapshot;
  classColors = class_colors;
  const ox = origin[0], oy = origin[1], oz = origin[2];
  const cellSize = (2 * extent) / resolution;
  _tmpScale.setScalar(cellSize * 0.7);
  const halfCell = cellSize / 2;
  // Dollhouse-cutaway cull: interior-facing voxels only. A voxel is "interior-
  // facing" if its surface normal points toward the chunk centroid.
  const cx = centroid?.[0] ?? 0, cy = centroid?.[1] ?? 1, cz = centroid?.[2] ?? 0;
  let culled = 0;

  let n = 0;
  for (let i = 0; i < cells.length && n < MAX_VOXELS; i++) {
    const row = cells[i];
    const ix = row[0], iy = row[1], iz = row[2], cls = row[3];

    // Sub-cell offset
    let dx = 0, dy = 0, dz = 0;
    if (row.length >= 11) {
      dx = (row[8] / 127) * halfCell;
      dy = (row[9] / 127) * halfCell;
      dz = (row[10] / 127) * halfCell;
    }
    const x = ox - extent + (ix + 0.5) * cellSize + dx;
    const y = oy - extent + (iy + 0.5) * cellSize + dy;
    const z = oz - extent + (iz + 0.5) * cellSize + dz;

    // Cutaway cull (only if normals present — backward compat with old snapshots)
    if (row.length >= 14) {
      const nx = row[11] / 127, ny = row[12] / 127, nz = row[13] / 127;
      const nmag2 = nx * nx + ny * ny + nz * nz;
      if (nmag2 > 0.01) {
        const dot = nx * (cx - x) + ny * (cy - y) + nz * (cz - z);
        if (dot < 0) { culled++; continue; }   // exterior face — skip
      }
    }

    // Per-voxel RGB if present, else class color
    let cr, cg, cb;
    if (row.length >= 8 && row[5] >= 0) {
      cr = row[5] / 255; cg = row[6] / 255; cb = row[7] / 255;
    } else {
      const hex = class_colors[cls] || '#ffffff';
      _tmpColor.set(hex);
      cr = _tmpColor.r; cg = _tmpColor.g; cb = _tmpColor.b;
    }
    cubesMesh.instanceColor.setXYZ(n, cr, cg, cb);

    // UVW: store per-voxel canonical coord on the instance so the
    // canonical-pass shader (matCanonical) can render each cube tinted by
    // its own (u, v, w) byte-identity. No hover-decode anymore.
    cubeUvwAttr.setXYZ(n, ix, iy, iz);

    _tmpMatrix.compose(new THREE.Vector3(x, y, z), _tmpQuat, _tmpScale);
    cubesMesh.setMatrixAt(n, _tmpMatrix);
    n++;
  }
  cubesMesh.count = n;
  cubesMesh.instanceColor.needsUpdate = true;
  cubesMesh.instanceMatrix.needsUpdate = true;
  cubeUvwAttr.needsUpdate = true;
  if (culled > 0) console.log(`[cutaway] kept ${n}, culled ${culled} exterior faces`);

  updateHud(snapshot);
  if (!classColorsLegendBuilt) buildLegend(class_colors, snapshot.class_names);
  if (snapshot.history) updateTrendChart(snapshot.history);
  if (snapshot.stats?.confidence_histogram) updateConfidenceChart(snapshot.stats.confidence_histogram);
  lastSnapshot = snapshot;
}

function renderGaussians(payload) {
  const arr = payload.gaussians || [];
  // Feed the cutaway-cull centroid uniform from the gauss cloud payload
  if (payload.centroid) {
    gaussMat.uniforms.uCentroid.value.set(
      payload.centroid[0], payload.centroid[1], payload.centroid[2]);
  }
  let n = 0;
  for (let i = 0; i < arr.length && n < MAX_GAUSS; i++) {
    const g = arr[i];
    const [x, y, z] = g.p;
    const [r, gC, b] = g.c;
    _tmpMatrix.compose(new THREE.Vector3(x, y, z), _tmpQuat, new THREE.Vector3(1, 1, 1));
    gaussMesh.setMatrixAt(n, _tmpMatrix);
    gInstColor.setXYZ(n, r, gC, b);
    gInstAlpha.setX(n, g.a);
    gInstScale.setX(n, g.s * 1.8);
    if (g.n) {
      gInstNormal.setXYZ(n, g.n[0], g.n[1], g.n[2]);
    } else {
      gInstNormal.setXYZ(n, 0, 0, 0);   // zero = "no normal" → shader skips cull
    }
    n++;
  }
  gaussMesh.count = n;
  gaussMesh.instanceMatrix.needsUpdate = true;
  gInstColor.needsUpdate = true;
  gInstAlpha.needsUpdate = true;
  gInstScale.needsUpdate = true;
  gInstNormal.needsUpdate = true;
  if (renderMode === 'gaussian') gaussMesh.visible = true;
  console.log(`[viewer] loaded ${n} gaussians (cutaway-cull active)`);
}

// ─── HUD ────────────────────────────────────────────────────────────────

let classColorsLegendBuilt = false;
function buildLegend(colors, names) {
  const el = document.getElementById('legend');
  el.innerHTML = '';
  for (const [cid, hex] of Object.entries(colors)) {
    if (cid === '0') continue;
    const row = document.createElement('div');
    row.className = 'row';
    const sw = document.createElement('div');
    sw.className = 'swatch';
    sw.style.background = hex;
    row.appendChild(sw);
    const label = document.createElement('span');
    label.textContent = names?.[cid] ?? `class ${cid}`;
    row.appendChild(label);
    el.appendChild(row);
  }
  classColorsLegendBuilt = true;
}

function updateHud(s) {
  document.getElementById('hud-iter').textContent = s.iteration;
  document.getElementById('hud-voxels').textContent = (s.stats?.active_voxels ?? 0).toLocaleString();
  document.getElementById('hud-converged').textContent = (s.stats?.convergence_pct ?? 0) + '%';
  document.getElementById('hud-meanconf').textContent = (s.stats?.mean_confidence ?? 0).toFixed(3);
  document.getElementById('hud-res').textContent = `${s.resolution}³ @ ±${s.extent}m`;

  const bar = document.getElementById('class-bar');
  bar.innerHTML = '';
  const counts = s.stats?.per_class_counts ?? {};
  for (const [cid, n] of Object.entries(counts)) {
    if (cid === '0') continue;
    const seg = document.createElement('div');
    seg.className = 'seg';
    seg.style.background = classColors[cid] || '#fff';
    seg.style.flex = `${n}`;
    seg.title = `${s.class_names?.[cid] ?? cid}: ${n}`;
    bar.appendChild(seg);
  }
}

function updateConfidenceChart(hist) {
  const cv = document.getElementById('confidence-chart');
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  const max = Math.max(1, ...hist);
  const bw = cv.width / hist.length;
  for (let i = 0; i < hist.length; i++) {
    const h = (hist[i] / max) * (cv.height - 4);
    // Color: red at 0% confidence, green at 100%
    const t = i / (hist.length - 1);
    const r = Math.round(232 * (1 - t) + 80 * t);
    const g = Math.round(90 * (1 - t) + 184 * t);
    const b = 80;
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(i * bw + 1, cv.height - h - 2, bw - 2, h);
  }
  // x-axis ticks: 0, 50, 100
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.font = '9px monospace';
  ctx.fillText('0%', 2, cv.height - 1);
  ctx.fillText('50', cv.width / 2 - 6, cv.height - 1);
  ctx.fillText('100%', cv.width - 22, cv.height - 1);
}

function updateTrendChart(history) {
  const cv = document.getElementById('trend-chart');
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (history.length < 2) return;

  const W = cv.width;
  const H = cv.height;
  const rates = history.map(h => h.change_rate);
  const maxR = Math.max(0.01, ...rates);

  // Convergence threshold reference line
  const TARGET = 0.02;
  const yTarget = H - 4 - (TARGET / maxR) * (H - 8);
  ctx.strokeStyle = 'rgba(255, 176, 112, 0.35)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(0, yTarget);
  ctx.lineTo(W, yTarget);
  ctx.stroke();
  ctx.setLineDash([]);

  // Change rate line
  ctx.strokeStyle = '#ffb070';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  history.forEach((h, i) => {
    const x = (i / Math.max(1, history.length - 1)) * (W - 4) + 2;
    const y = H - 4 - (h.change_rate / maxR) * (H - 8);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Latest value annotated
  const last = history[history.length - 1];
  ctx.fillStyle = '#ffb070';
  ctx.font = '10px monospace';
  ctx.fillText(`Δ ${last.change_rate.toFixed(4)}`, 4, 11);
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.fillText(`tol ${TARGET}`, W - 50, yTarget - 2);
  ctx.fillText(`i${history[0].iter}`, 2, H - 1);
  ctx.fillText(`i${last.iter}`, W - 22, H - 1);
}

// ─── WebSocket ──────────────────────────────────────────────────────────

let ws = null;
let reconnectTimer = null;
function connect() {
  const status = document.getElementById('status');
  status.textContent = 'connecting…';
  status.className = '';
  try {
    ws = new WebSocket(WS_URL);
  } catch (e) {
    status.textContent = 'connect failed: ' + e.message;
    status.className = 'error';
    return scheduleReconnect();
  }
  ws.onopen = () => {
    status.textContent = `connected ${WS_URL}`;
    status.className = 'connected';
  };
  ws.onmessage = async ev => {
    try {
      const data = JSON.parse(ev.data);
      if (data.type === 'voxel_snapshot') {
        renderVoxels(data);
      } else if (data.type === 'phase_b_complete') {
        document.getElementById('status').textContent =
          `phase B complete · ${data.n_gaussians} gaussians ready (press GAUSSIAN above)`;
        // Auto-load the gaussians file
        try {
          const g = await fetch(data.gaussians_url).then(r => r.json());
          renderGaussians(g);
        } catch (e) {
          console.warn('failed to load gaussians:', e);
        }
      }
    } catch (e) {
      console.error('bad message:', e);
    }
  };
  ws.onclose = () => {
    status.textContent = 'disconnected — retrying…';
    status.className = 'error';
    scheduleReconnect();
  };
  ws.onerror = () => {};
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => { reconnectTimer = null; connect(); }, 2000);
}
connect();

// ─── Animation loop ─────────────────────────────────────────────────────
// Hover-decode was removed: UVW mode is now a pure visual rendering toggle
// — cubes are colored by their own (u, v, w) byte-identity via matCanonical.
// One render pass per frame regardless of mode.

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

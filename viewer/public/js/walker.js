// Walker — input + movement on a walkable polygon, scene-transition triggering.
//
// Flat-screen mode:  WASD or arrows move the hero. Mouse rotates camera around
//                    the hero (FF7/FF9 style fixed-distance follow).
// VR mode:           Left thumbstick moves the rig (you walk). Right stick turns.
// AR/passthrough:    Same as VR but rig sits at your real-world floor anchor.
//
// The hero mesh itself follows the rig in VR (so you see your own character in
// front of you) or is the player avatar in flat mode (third-person).

import * as THREE from 'three';

const WALK_SPEED = 2.8;       // m/s
const TURN_SPEED = 2.0;       // rad/s
const HERO_FOLLOW_DIST = 5.5; // m, third-person camera distance in flat mode
const HERO_FOLLOW_HEIGHT = 2.8;

export class Walker {
  constructor({ scene, camera, rig, renderer, gltfLoader }) {
    this.scene = scene;
    this.camera = camera;
    this.rig = rig;
    this.renderer = renderer;
    this.gltfLoader = gltfLoader;

    this.hero = this._buildHeroPlaceholder();
    this.hero.position.set(0, 0, 0);
    scene.add(this.hero);

    this.position = new THREE.Vector3(0, 0, 0);
    this.facing = 0;                  // radians, 0 = facing -Z
    this.cameraYaw = 0;               // camera orbit around hero in flat mode
    this.cameraPitch = 0.25;

    this._keys = new Set();
    this._mouseButtons = 0;
    this._lastMouse = { x: 0, y: 0 };
    this._walkable = null;
    this._transitionHandler = null;
    this._interactHandler = null;
    this._inTransition = false;
    this._lastInteractScan = 0;

    this._bindInput();
    this._setupXRControllers();
  }

  setWalkable(walkable) { this._walkable = walkable; }
  setSceneTransitionHandler(fn) { this._transitionHandler = fn; }
  setActorInteractHandler(fn) { this._interactHandler = fn; }

  async loadHeroModel(manifest) {
    const hero = manifest?.characters?.hero;
    if (!hero?.model || !this.gltfLoader) return;
    try {
      const gltf = await new Promise((resolve, reject) =>
        this.gltfLoader.load(`./${hero.model}`, resolve, undefined, reject));
      // Replace placeholder hero with loaded model, preserving transform
      const newHero = gltf.scene;
      // Hunyuan3D meshes come in roughly unit-box space; scale to hero.scaleMeters
      const box = new THREE.Box3().setFromObject(newHero);
      const size = new THREE.Vector3();
      box.getSize(size);
      const targetHeight = hero.scaleMeters || 1.75;
      const scale = targetHeight / Math.max(0.001, size.y);
      newHero.scale.setScalar(scale);
      // Recenter so feet are at y=0
      const center = new THREE.Vector3();
      box.getCenter(center);
      newHero.position.set(-center.x * scale, -box.min.y * scale, -center.z * scale);
      // Wrap so we can move the outer group while keeping the inner offsets
      const wrapper = new THREE.Group();
      wrapper.add(newHero);
      wrapper.position.copy(this.hero.position);
      wrapper.rotation.y = this.facing;
      this.scene.add(wrapper);
      this.scene.remove(this.hero);
      this.hero.traverse(o => { if (o.isMesh) { o.geometry?.dispose(); o.material?.dispose?.(); } });
      this.hero = wrapper;
    } catch (e) {
      console.warn('Failed to load hero model, keeping placeholder:', e);
    }
  }

  spawnAt(manifest, sceneId, spawnId) {
    const def = manifest.scenes[sceneId];
    const spawn = def?.spawns?.[spawnId] ?? { position: [0, 0, 0], facing: 0 };
    this.position.set(spawn.position[0], spawn.position[1], spawn.position[2]);
    this.facing = (spawn.facing ?? 0) * Math.PI / 180;
    this._applyHeroTransform();
    this._inTransition = false;
  }

  update(t, frame) {
    if (this._inTransition) return;

    const xrActive = this.renderer.xr.isPresenting;

    // Movement intent (X/Z in scene space)
    let mx = 0, mz = 0;
    let turn = 0;

    if (xrActive) {
      // Read controller thumbsticks
      const session = this.renderer.xr.getSession();
      for (const src of (session?.inputSources ?? [])) {
        if (!src.gamepad) continue;
        const axes = src.gamepad.axes;
        if (src.handedness === 'left' && axes.length >= 4) {
          // Left stick: walk
          const ax = axes[2] ?? axes[0];
          const ay = axes[3] ?? axes[1];
          if (Math.abs(ax) > 0.15) mx += ax;
          if (Math.abs(ay) > 0.15) mz += ay;
        }
        if (src.handedness === 'right' && axes.length >= 4) {
          const ax = axes[2] ?? axes[0];
          if (Math.abs(ax) > 0.15) turn += ax;
        }
      }
    } else {
      if (this._keys.has('KeyW') || this._keys.has('ArrowUp'))    mz -= 1;
      if (this._keys.has('KeyS') || this._keys.has('ArrowDown'))  mz += 1;
      if (this._keys.has('KeyA') || this._keys.has('ArrowLeft'))  mx -= 1;
      if (this._keys.has('KeyD') || this._keys.has('ArrowRight')) mx += 1;
    }

    // Translate intent through current camera-yaw so "forward" is toward the screen
    const yaw = xrActive ? this._getXRYaw() : this.cameraYaw;
    const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
    const worldDX = mx * cosY + mz * sinY;
    const worldDZ = -mx * sinY + mz * cosY;

    const moving = Math.abs(mx) + Math.abs(mz) > 0.01;
    if (moving) {
      this.facing = Math.atan2(worldDX, worldDZ);
      const dt = Math.min(0.05, this._dt(t));
      const nx = this.position.x + worldDX * WALK_SPEED * dt;
      const nz = this.position.z + worldDZ * WALK_SPEED * dt;
      if (this._isWalkable(nx, nz)) {
        this.position.x = nx;
        this.position.z = nz;
      } else {
        // Try axis-separated slide
        if (this._isWalkable(nx, this.position.z)) this.position.x = nx;
        else if (this._isWalkable(this.position.x, nz)) this.position.z = nz;
      }
    }

    if (Math.abs(turn) > 0.01) {
      this.cameraYaw -= turn * TURN_SPEED * this._dt(t);
    }

    this._applyHeroTransform();

    if (xrActive) {
      // Rig follows hero so the user's head is anchored above the hero's position
      this.rig.position.set(this.position.x, this.position.y, this.position.z);
      this.rig.rotation.y = this.cameraYaw;
    } else {
      this._followCamera();
    }

    // Check scene exits (every frame)
    this._checkExits();

    // Check actor interactions (throttled — keystroke or controller-button)
    if (t - this._lastInteractScan > 0.2 && this._consumeInteractIntent()) {
      this._lastInteractScan = t;
      this._tryInteract();
    }

    this._lastT = t;
  }

  _dt(t) {
    if (this._lastT == null) { this._lastT = t; return 1 / 60; }
    return t - this._lastT;
  }

  _applyHeroTransform() {
    this.hero.position.copy(this.position);
    this.hero.rotation.y = this.facing;
  }

  _followCamera() {
    const yaw = this.cameraYaw;
    const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
    const offX = HERO_FOLLOW_DIST * sinY;
    const offZ = HERO_FOLLOW_DIST * cosY;
    this.camera.position.set(this.position.x + offX, this.position.y + HERO_FOLLOW_HEIGHT, this.position.z + offZ);
    this.camera.lookAt(this.position.x, this.position.y + 1.0, this.position.z);
  }

  _getXRYaw() {
    // Approximate yaw of the headset
    const cam = this.renderer.xr.getCamera();
    const euler = new THREE.Euler().setFromQuaternion(cam.quaternion, 'YXZ');
    return euler.y + this.cameraYaw;
  }

  _isWalkable(x, z) {
    if (!this._walkable?.polygons?.length) return true;
    for (const poly of this._walkable.polygons) {
      if (pointInPolygon([x, z], poly)) return true;
    }
    return false;
  }

  _checkExits() {
    if (!this._walkable?.exits?.length) return;
    for (const exit of this._walkable.exits) {
      if (exit.trigger?.type === 'polygon') {
        if (pointInPolygon([this.position.x, this.position.z], exit.trigger.points)) {
          this._beginTransition(exit.destination);
          return;
        }
      }
    }
  }

  async _beginTransition(destination) {
    if (this._inTransition) return;
    this._inTransition = true;
    if (this._transitionHandler) await this._transitionHandler(destination);
  }

  _tryInteract() {
    // Find the nearest actor within ~1.6m in front of the hero
    const facingX = Math.sin(this.facing);
    const facingZ = Math.cos(this.facing);
    let best = null, bestD = Infinity;
    this.scene.traverse(o => {
      if (!o.userData?.isActor) return;
      const dx = o.position.x - this.position.x;
      const dz = o.position.z - this.position.z;
      const d = Math.hypot(dx, dz);
      if (d > 1.6) return;
      // Must be roughly in front
      const dot = (dx * facingX + dz * facingZ) / Math.max(0.01, d);
      if (dot < 0.3) return;
      if (d < bestD) { bestD = d; best = o; }
    });
    if (best && this._interactHandler) this._interactHandler({ character: best.userData.character, object: best });
  }

  _consumeInteractIntent() {
    if (this._keys.has('Space') || this._keys.has('KeyE') || this._keys.has('Enter')) {
      this._keys.delete('Space'); this._keys.delete('KeyE'); this._keys.delete('Enter');
      return true;
    }
    if (this._xrInteractIntent) { this._xrInteractIntent = false; return true; }
    return false;
  }

  _setupXRControllers() {
    this._xrInteractIntent = false;
    const session = () => this.renderer.xr.getSession();
    this.renderer.xr.addEventListener('sessionstart', () => {
      const s = session();
      if (!s) return;
      s.addEventListener('select', () => { this._xrInteractIntent = true; });
    });
  }

  _bindInput() {
    addEventListener('keydown', e => { this._keys.add(e.code); });
    addEventListener('keyup',   e => { this._keys.delete(e.code); });
    const dom = this.renderer.domElement;
    dom.addEventListener('mousedown', e => {
      this._mouseButtons |= (1 << e.button);
      this._lastMouse = { x: e.clientX, y: e.clientY };
    });
    addEventListener('mouseup',   e => { this._mouseButtons &= ~(1 << e.button); });
    addEventListener('mousemove', e => {
      if (this._mouseButtons & 1) {
        const dx = e.clientX - this._lastMouse.x;
        this.cameraYaw -= dx * 0.005;
        this._lastMouse = { x: e.clientX, y: e.clientY };
      }
    });
  }

  _buildHeroPlaceholder() {
    const g = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.25, 1.1, 6, 12),
      new THREE.MeshStandardMaterial({ color: 0x6a8aaa, roughness: 0.6 })
    );
    body.position.y = 0.85;
    g.add(body);
    const head = new THREE.Mesh(
      new THREE.SphereGeometry(0.18, 16, 12),
      new THREE.MeshStandardMaterial({ color: 0xe8c4a0, roughness: 0.5 })
    );
    head.position.y = 1.65;
    g.add(head);
    const nose = new THREE.Mesh(
      new THREE.BoxGeometry(0.05, 0.05, 0.12),
      new THREE.MeshStandardMaterial({ color: 0x402010 })
    );
    nose.position.set(0, 1.65, 0.18);
    g.add(nose);
    return g;
  }
}

function pointInPolygon([x, y], poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    const intersect = ((yi > y) !== (yj > y)) &&
                      (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

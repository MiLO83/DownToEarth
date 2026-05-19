// SceneLoader — fetches a scene's mesh + walkable polygon, manages actors,
// and replaces lighting between scene transitions.

import * as THREE from 'three';

export class SceneLoader {
  constructor({ scene, gltfLoader }) {
    this.scene = scene;
    this.loader = gltfLoader;
    this.currentRoot = null;
    this.actors = new Map();   // character id → THREE.Object3D
    this.walkable = null;       // { polygons: [[ [x,z], ... ]], exits: [...] }
    this.cameraFraming = null;
    this.activeSceneId = null;
  }

  async loadScene(manifest, sceneId) {
    const def = manifest.scenes[sceneId];
    if (!def) throw new Error('No such scene: ' + sceneId);

    // Tear down previous scene root
    if (this.currentRoot) {
      this.scene.remove(this.currentRoot);
      this.currentRoot.traverse(o => { if (o.isMesh) { o.geometry?.dispose(); o.material?.dispose?.(); } });
    }
    this.actors.clear();

    const root = new THREE.Group();
    root.name = `scene:${sceneId}`;
    this.scene.add(root);
    this.currentRoot = root;
    this.activeSceneId = sceneId;
    this.cameraFraming = def.cameraFraming;

    // Lights
    for (const L of (def.lights || [])) {
      if (L.type === 'directional') {
        const light = new THREE.DirectionalLight(new THREE.Color(L.color), L.intensity ?? 1);
        const d = L.direction ?? [0, -1, 0];
        light.position.set(-d[0] * 10, -d[1] * 10, -d[2] * 10);
        light.target.position.set(0, 0, 0);
        root.add(light, light.target);
      } else if (L.type === 'ambient') {
        root.add(new THREE.AmbientLight(new THREE.Color(L.color), L.intensity ?? 0.3));
      }
    }

    // Scene mesh — may not exist yet if user hasn't run pipeline; fall back to placeholder
    if (def.mesh) {
      try {
        const url = `./${def.mesh}`;
        const gltf = await this.loadGLTF(url);
        gltf.scene.traverse(o => {
          if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; }
        });
        root.add(gltf.scene);
      } catch (e) {
        console.warn(`Scene mesh '${def.mesh}' missing — using placeholder ground`);
        root.add(this._placeholderGround(def.ambientColor));
      }
    } else {
      root.add(this._placeholderGround(def.ambientColor));
    }

    // Walkable polygon
    if (def.walkable) {
      try {
        const wRes = await fetch(`./${def.walkable}`);
        if (wRes.ok) this.walkable = await wRes.json();
        else this.walkable = this._defaultWalkable();
      } catch {
        this.walkable = this._defaultWalkable();
      }
    } else {
      this.walkable = this._defaultWalkable();
    }
    this.walkable.exits = def.exits || [];

    // Actors
    for (const a of (def.actors || [])) {
      const cdef = manifest.characters[a.character];
      if (!cdef) continue;
      let obj;
      if (cdef.model) {
        try {
          const gltf = await this.loadGLTF(`./${cdef.model}`);
          obj = gltf.scene;
        } catch {
          obj = this._placeholderActor(cdef.displayName);
        }
      } else {
        obj = this._placeholderActor(cdef.displayName);
      }
      obj.position.set(a.position[0], a.position[1], a.position[2]);
      obj.rotation.y = (a.facing ?? 0) * Math.PI / 180;
      obj.userData.character = a.character;
      obj.userData.isActor = true;
      root.add(obj);
      this.actors.set(a.character, obj);
    }
  }

  loadGLTF(url) {
    return new Promise((resolve, reject) => {
      this.loader.load(url, resolve, undefined, reject);
    });
  }

  _placeholderGround(color = '#9a8a6a') {
    const g = new THREE.Group();
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(20, 20),
      new THREE.MeshStandardMaterial({ color: new THREE.Color(color), roughness: 0.95 })
    );
    plane.rotation.x = -Math.PI / 2;
    plane.receiveShadow = true;
    g.add(plane);
    // Cardinal markers so you can tell which way you're facing in placeholder mode
    const dirColors = ['#ff6060', '#60ff60', '#6060ff', '#ffff60'];
    const dirPositions = [[0, 0.05, -8], [8, 0.05, 0], [0, 0.05, 8], [-8, 0.05, 0]];
    for (let i = 0; i < 4; i++) {
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(0.4, 0.1, 0.4),
        new THREE.MeshStandardMaterial({ color: new THREE.Color(dirColors[i]) })
      );
      m.position.set(...dirPositions[i]);
      g.add(m);
    }
    return g;
  }

  _placeholderActor(name) {
    const g = new THREE.Group();
    // Body
    const body = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.25, 1.1, 4, 8),
      new THREE.MeshStandardMaterial({ color: 0xc88a5a, roughness: 0.7 })
    );
    body.position.y = 0.85;
    g.add(body);
    // Head
    const head = new THREE.Mesh(
      new THREE.SphereGeometry(0.18, 12, 10),
      new THREE.MeshStandardMaterial({ color: 0xe0b890, roughness: 0.6 })
    );
    head.position.y = 1.65;
    g.add(head);
    // Facing indicator (small block on front of head)
    const nose = new THREE.Mesh(
      new THREE.BoxGeometry(0.05, 0.05, 0.1),
      new THREE.MeshStandardMaterial({ color: 0x202020 })
    );
    nose.position.set(0, 1.65, 0.18);
    g.add(nose);
    return g;
  }

  _defaultWalkable() {
    return {
      polygons: [[[-8, -8], [8, -8], [8, 8], [-8, 8]]],
      exits: []
    };
  }
}

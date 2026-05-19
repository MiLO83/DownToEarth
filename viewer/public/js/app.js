// Down To Earth — WebXR walker
//
// Loads a manifest of JRPG-style scenes (each a 3D mesh + walkable polygon)
// and lets the player walk between them as a 3D character. Supports flat,
// immersive VR, and AR passthrough modes on Quest 3 browser.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { ARButton } from 'three/addons/webxr/ARButton.js';
import { Walker } from './walker.js';
import { SceneLoader } from './scene-loader.js';
import { Dialog } from './dialog.js';

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.xr.enabled = true;
document.getElementById('app').appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.05, 200);
camera.position.set(0, 3.2, 6.5);
camera.lookAt(0, 1.0, 0);

// Camera rig for XR — we move the rig, the XR camera stays at its measured pose
const rig = new THREE.Group();
rig.add(camera);
scene.add(rig);

const gltfLoader = new GLTFLoader();
const sceneLoader = new SceneLoader({ scene, gltfLoader });
const walker = new Walker({ scene, camera, rig, renderer, gltfLoader });
const dialog = new Dialog(document.getElementById('dialog'));

// XR session buttons
const vrButton = VRButton.createButton(renderer);
vrButton.id = 'vr-btn-internal';
vrButton.style.display = 'none';
document.body.appendChild(vrButton);
document.getElementById('vr-btn').addEventListener('click', () => vrButton.click());

if ('xr' in navigator) {
  navigator.xr.isSessionSupported('immersive-ar').then(supported => {
    if (!supported) return;
    const arButton = ARButton.createButton(renderer, {
      requiredFeatures: ['local-floor'],
      optionalFeatures: ['hand-tracking', 'plane-detection']
    });
    arButton.style.display = 'none';
    document.body.appendChild(arButton);
    const visibleAr = document.getElementById('ar-btn');
    visibleAr.style.display = '';
    visibleAr.addEventListener('click', () => arButton.click());
  });
}

// Resize
addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});

// Boot
(async () => {
  const manifestRes = await fetch('./manifest.json');
  const manifest = await manifestRes.json();
  window.__manifest = manifest;

  await sceneLoader.loadScene(manifest, manifest.startScene);
  await walker.loadHeroModel(manifest);
  walker.setWalkable(sceneLoader.walkable);
  walker.spawnAt(manifest, manifest.startScene, manifest.startSpawn);
  walker.setSceneTransitionHandler(async (destination) => {
    await sceneLoader.loadScene(manifest, destination.scene);
    walker.setWalkable(sceneLoader.walkable);
    walker.spawnAt(manifest, destination.scene, destination.spawn);
    document.getElementById('scene-label').textContent =
      manifest.scenes[destination.scene]?.displayName ?? destination.scene;
  });
  walker.setActorInteractHandler((actor) => {
    const cdef = manifest.characters[actor.character];
    dialog.show(cdef.displayName, sampleDialogFor(actor.character));
  });
  document.getElementById('scene-label').textContent =
    manifest.scenes[manifest.startScene]?.displayName ?? manifest.startScene;
  document.getElementById('loading').classList.add('hidden');
})().catch(e => {
  console.error('Boot failed:', e);
  document.getElementById('loading').textContent = 'Failed: ' + e.message;
});

renderer.setAnimationLoop((time, frame) => {
  walker.update(time / 1000, frame);
  renderer.render(scene, camera);
});

// Tiny placeholder dialog source — replace with manifest-driven content later
function sampleDialogFor(charId) {
  return ({
    'elder-bren': 'You walk lightly, traveler. The shrine in the east woods still pulses some nights. Mind the path.',
    'merchant-pip': 'Got nothing flashy today — but the road sells its own stories, eh?'
  })[charId] ?? '...';
}

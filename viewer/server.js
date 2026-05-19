// Tiny static server for the DownToEarth viewer.
// Serves /public on http://localhost:5173.
// Use HTTPS in production — WebXR requires it for non-localhost.

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 5173;
const ROOT = path.resolve(__dirname, 'public');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'application/javascript',
  '.json': 'application/json',
  '.glb':  'model/gltf-binary',
  '.gltf': 'model/gltf+json',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.svg':  'image/svg+xml',
  '.css':  'text/css',
  '.mp3':  'audio/mpeg',
  '.ogg':  'audio/ogg',
  '.wasm': 'application/wasm',
  '.map':  'application/json',
};

http.createServer((req, res) => {
  let urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
  if (urlPath === '/' || urlPath === '') urlPath = '/index.html';
  let fp = path.normalize(path.join(ROOT, urlPath));
  if (!fp.startsWith(ROOT)) { res.writeHead(403); return res.end('forbidden'); }
  if (!fs.existsSync(fp) || fs.statSync(fp).isDirectory()) {
    res.writeHead(404);
    return res.end('not found: ' + urlPath);
  }
  const ct = MIME[path.extname(fp).toLowerCase()] || 'application/octet-stream';
  res.writeHead(200, {
    'Content-Type': ct,
    'Content-Length': fs.statSync(fp).size,
    'Cache-Control': 'no-cache',
    // Required headers for WebXR + cross-origin isolation when streaming large GLBs
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cross-Origin-Opener-Policy': 'same-origin',
  });
  fs.createReadStream(fp).pipe(res);
}).listen(PORT, () => {
  console.log(`Down To Earth viewer  →  http://localhost:${PORT}`);
  console.log(`(For Quest 3:  point Meta Browser at  http://<this-machine-ip>:${PORT})`);
});

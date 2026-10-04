// Serves a built site folder under a sub-path, the way GitHub Pages serves a project site: plain
// static files, no cross-origin-isolation headers, a 404.html for unknown paths.
//
//   node scripts/serve-site.mjs [dir=_site] [prefix=/parts-sim/] [port=4180]
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.svg': 'image/svg+xml', '.jpg': 'image/jpeg',
  '.png': 'image/png', '.wasm': 'application/wasm', '.txt': 'text/plain; charset=utf-8',
};

export function serveSite(dir, prefix = '/parts-sim/', port = 0) {
  const base = resolve(dir);
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://x');
    let rel = decodeURIComponent(url.pathname);
    const send = async (file, status = 200) => {
      const body = await readFile(file);
      res.writeHead(status, { 'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream', 'Cache-Control': 'no-cache' });
      res.end(body);
    };
    if (!rel.startsWith(prefix)) { res.writeHead(404); res.end('outside the site prefix'); return; }
    rel = normalize(rel.slice(prefix.length));
    let file = join(base, rel);
    if (!file.startsWith(base)) { res.writeHead(403); res.end(); return; }
    try {
      const st = await stat(file);
      if (st.isDirectory()) {
        if (!url.pathname.endsWith('/')) { res.writeHead(301, { Location: url.pathname + '/' + url.search }); res.end(); return; }
        file = join(file, 'index.html');
      }
      await send(file);
    } catch {
      try { await send(join(base, '404.html'), 404); } catch { res.writeHead(404); res.end('not found'); }
    }
  });
  return new Promise((ok) => server.listen(port, '127.0.0.1', () => ok({ server, url: `http://127.0.0.1:${server.address().port}${prefix}` })));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { url } = await serveSite(process.argv[2] ?? '_site', process.argv[3] ?? '/parts-sim/', Number(process.argv[4] ?? 4180));
  console.log(`serving at ${url}`);
}

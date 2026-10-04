// The download page: puts the visitor's system first, and asks GitHub for the latest release so the
// buttons follow what exists now. If that fails (offline, rate limit), the page keeps the cards that
// were built into it from downloads.json.
import { REPO, downloadsHtml, fromRelease } from './downloads-lib.mjs';

const list = document.getElementById('dl-list');
const platform = () => {
  const p = (navigator.userAgentData?.platform || navigator.platform || '') + ' ' + navigator.userAgent;
  if (/android|iphone|ipad|ipod/i.test(p)) return null;
  if (/win/i.test(p)) return 'windows';
  if (/mac/i.test(p)) return 'mac';
  if (/linux|x11|cros/i.test(p)) return 'linux';
  return null;
};

function order() {
  const mine = platform();
  const box = document.getElementById('dl-list');
  if (!box) return;
  for (const s of box.querySelectorAll('.os')) {
    const yours = s.dataset.os === mine;
    s.style.order = yours ? '0' : '1';
    s.querySelector('.yours').hidden = !yours;
  }
}

function show(data, from) {
  const holder = document.getElementById('dl-holder');
  holder.innerHTML = downloadsHtml(data);
  document.getElementById('dl-source').textContent = from;
  order();
}

order();
try {
  const r = await fetch(`https://api.github.com/repos/${REPO}/releases/latest`, { headers: { Accept: 'application/vnd.github+json' } });
  if (r.ok) {
    const data = fromRelease(await r.json());
    if (data.assets.length) show(data, `Checked just now against GitHub's latest release (${data.tag}).`);
  }
} catch { /* keep the built-in cards */ }

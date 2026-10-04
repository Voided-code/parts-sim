// Runs in the web app on the website. Without cross-origin isolation the solvers cannot share memory
// between threads, so they run on one thread; say so instead of just being slow.
(() => {
  const check = () => {
    if (self.crossOriginIsolated) return;
    const bar = document.createElement('div');
    bar.setAttribute('role', 'status');
    bar.textContent = 'Running on one thread: this browser did not allow the shared memory the helper threads need, so solves are slower. A current Chrome, Edge or Firefox over https usually allows it.';
    bar.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:99999;padding:6px 12px;background:#fff4d6;color:#14181f;border-top:1px solid #d9a400;font:13px system-ui,sans-serif';
    const close = document.createElement('button');
    close.textContent = 'Dismiss';
    close.style.cssText = 'margin-left:12px';
    close.onclick = () => bar.remove();
    bar.append(close);
    document.body.append(bar);
  };
  // the service worker may still be taking over (it reloads the page once); look after that settles
  addEventListener('load', () => setTimeout(check, 1500));
})();

// Small page enhancements. The pages work without this file.
const $ = (s, r = document) => r.querySelector(s);

// Landing page: say whether this browser can run the web app's GPU path.
const gpu = $('#gpu-note');
if (gpu) {
  if (!('gpu' in navigator)) {
    gpu.hidden = false;
    gpu.textContent = 'This browser has no WebGPU. The web app still opens, but airflow runs on the CPU with a coarse grid (up to 1 million cells), and the structural solvers use the CPU too. Chrome or Edge 113 or newer, or Safari 26 or newer, give the GPU path.';
  } else {
    navigator.gpu.requestAdapter().then((a) => {
      if (!a) {
        gpu.hidden = false;
        gpu.textContent = 'This browser reports WebGPU but found no usable graphics adapter, so the web app falls back to the CPU: airflow stops at a coarse grid of 1 million cells.';
      }
    }, () => {});
  }
}

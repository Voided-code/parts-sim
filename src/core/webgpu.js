// Shared WebGPU adapter lookup for the GPU solvers.

/**
 * The machine's graphics adapter, or null when WebGPU is unavailable. Software fallback adapters
 * (SwiftShader, WARP) are refused: they emulate a GPU on the CPU, compile the solver shaders slowly
 * and would be reported as the GPU, so the CPU engines serve those machines instead.
 */
export async function hardwareAdapter(options = {}) {
  if (typeof navigator === 'undefined' || !navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter(options);
  if (!adapter || adapter.info?.isFallbackAdapter || adapter.isFallbackAdapter) return null;
  return adapter;
}

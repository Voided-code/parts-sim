// Bridge between the sandboxed app page and the desktop shell. Exposes only these calls.
const { contextBridge, ipcRenderer, webUtils } = require('electron');

contextBridge.exposeInMainWorld('partsSimDesktop', {
  platform: process.platform,
  /** Tell the shell the page can receive files. */
  ready: () => ipcRenderer.send('renderer-ready'),
  /** Show the native Open dialog; chosen files arrive through onOpenFiles. */
  openDialog: () => ipcRenderer.send('open-dialog'),
  /** files: [{ name, path, data: Uint8Array }] */
  onOpenFiles: (cb) => ipcRenderer.on('open-files', (_e, files) => cb(files)),
  /** Menu commands such as 'view:iso', 'run:bend', 'sample:beam'. */
  onCommand: (cb) => ipcRenderer.on('command', (_e, command) => cb(command)),
  /** Absolute path of a dropped File (empty for files that did not come from disk). */
  pathForFile: (file) => webUtils.getPathForFile(file),
  /** Bytes of a .SLDPRT sitting next to an opened assembly, or null. */
  readSibling: (openedPath, name) => ipcRenderer.invoke('read-sibling', openedPath, name),
  /** Native save dialog for a PNG data URL. */
  saveImage: (dataUrl, name) => ipcRenderer.invoke('save-image', dataUrl, name),
  /** Native save dialog for an exported STL (bytes). */
  saveFile: (data, name) => ipcRenderer.invoke('save-file', data, name),
});

// benchmark mode only (--bench): the page saves its report through the shell
if (process.argv.includes('--parts-sim-bench')) {
  contextBridge.exposeInMainWorld('partsSimBench', {
    write: (json) => ipcRenderer.invoke('bench-write', json),
    readNative: () => ipcRenderer.invoke('bench-native'),
    done: () => ipcRenderer.send('bench-done'),
  });
}

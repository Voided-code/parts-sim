// electron-builder afterPack hook. Folders synced by iCloud Drive or other file providers add
// extended attributes (Finder info, provenance) that make `codesign` refuse the app bundle.
// Strip them from the packed app before it is signed.
const { execFileSync } = require('node:child_process');

exports.default = async function afterPack(context) {
  if (context.electronPlatformName !== 'darwin') return;
  execFileSync('xattr', ['-cr', context.appOutDir]);
};

// Starts the desktop app from the repo (after `vite build`). Clears ELECTRON_RUN_AS_NODE,
// which editor terminals (VS Code and other Electron apps) can leak into child processes
// and which would make Electron start as plain Node.
import { spawn } from 'node:child_process';
import electron from 'electron';

const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;
const child = spawn(electron, ['.', ...process.argv.slice(2)], { stdio: 'inherit', env });
child.on('exit', (code) => process.exit(code ?? 0));

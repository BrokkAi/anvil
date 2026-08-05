#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const repositoryDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const result = spawnSync('npm', ['run', 'generate'], {
  cwd: path.join(repositoryDir, 'sdk', 'typescript'),
  stdio: 'inherit',
});
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);

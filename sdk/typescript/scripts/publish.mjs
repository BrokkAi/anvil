#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { readFile, rm } from 'node:fs/promises';
import { mkdtempSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const packageDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const packageJson = JSON.parse(await readFile(path.join(packageDir, 'package.json'), 'utf8'));
const args = process.argv.slice(2);
const tarballIndex = args.indexOf('--tarball');
if (tarballIndex < 0 || !args[tarballIndex + 1]) {
  throw new Error('usage: publish.mjs --tarball PATH [--yes-publish]');
}
const tarball = path.resolve(args[tarballIndex + 1]);
const publish = args.includes('--yes-publish');

function npm(command, options = {}) {
  const result = spawnSync('npm', command, { encoding: 'utf8', ...options });
  if (result.error) throw result.error;
  return result;
}

function versionExists() {
  const result = npm(['view', `${packageJson.name}@${packageJson.version}`, 'version', '--json']);
  if (result.status === 0) return JSON.parse(result.stdout.trim()) === packageJson.version;
  if (/E404/.test(result.stderr)) return false;
  throw new Error(`npm view failed:\n${result.stderr}`);
}

async function waitUntilVisible() {
  for (let attempt = 1; attempt <= 20; attempt += 1) {
    if (versionExists()) return;
    process.stdout.write(`  waiting for registry visibility (${attempt}/20)\n`);
    await new Promise((resolve) => setTimeout(resolve, 15_000));
  }
  throw new Error(`${packageJson.name}@${packageJson.version} did not become visible`);
}

if (versionExists()) {
  process.stdout.write(`${packageJson.name}@${packageJson.version} already published; skipping\n`);
  process.exit(0);
}
if (!publish) {
  process.stdout.write(`would publish ${packageJson.name}@${packageJson.version} from ${tarball}\n`);
  process.exit(0);
}

const published = npm(['publish', tarball, '--access', 'public', '--provenance'], { stdio: 'inherit' });
if (published.status !== 0) throw new Error('npm publish failed');
await waitUntilVisible();

const temporaryDir = mkdtempSync(path.join(os.tmpdir(), 'anvil-sdk-npm-'));
try {
  const installed = npm(
    ['install', '--ignore-scripts', '--prefix', temporaryDir, `${packageJson.name}@${packageJson.version}`],
    { env: { ...process.env, npm_config_cache: path.join(temporaryDir, 'cache') } },
  );
  if (installed.status !== 0) throw new Error(`clean install failed:\n${installed.stderr}`);
  const imported = spawnSync(
    process.execPath,
    [
      '--input-type=module',
      '--eval',
      `import('${packageJson.name}').then((m) => { if (typeof m.AnvilClient !== 'function') process.exit(1) })`,
    ],
    { cwd: temporaryDir, encoding: 'utf8' },
  );
  if (imported.status !== 0) throw new Error(`clean import failed:\n${imported.stderr}`);
} finally {
  await rm(temporaryDir, { recursive: true, force: true });
}

process.stdout.write(`${packageJson.name}@${packageJson.version} published and verified\n`);

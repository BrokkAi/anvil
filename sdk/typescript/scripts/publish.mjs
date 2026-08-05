#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { readFile, rm } from 'node:fs/promises';
import { mkdtempSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { versionExists, waitUntilVisible } from '../../../npm/lib/registry.mjs';

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

function verifyTarballMetadata() {
  const result = npm(['publish', tarball, '--dry-run', '--json', '--ignore-scripts']);
  if (result.status !== 0) throw new Error(`npm publish dry run failed:\n${result.stderr}`);
  const report = JSON.parse(result.stdout);
  const metadata = report[packageJson.name] ?? report;
  if (metadata.name !== packageJson.name || metadata.version !== packageJson.version) {
    throw new Error(
      `tarball identity mismatch: expected ${packageJson.name}@${packageJson.version}, got ${metadata.name}@${metadata.version}`,
    );
  }
}

// The existence check comes first: `npm publish --dry-run` also talks to the
// registry and fails outright with "cannot publish over the previously
// published versions", so re-running after a successful publication has to
// take the skip path before the tarball is offered to the registry at all.
if (versionExists(packageJson.name, packageJson.version)) {
  process.stdout.write(`${packageJson.name}@${packageJson.version} already published; skipping\n`);
  process.exit(0);
}

verifyTarballMetadata();

if (!publish) {
  process.stdout.write(`would publish ${packageJson.name}@${packageJson.version} from ${tarball}\n`);
  process.exit(0);
}

const result = npm(['publish', tarball, '--access', 'public', '--provenance'], { stdio: 'inherit' });
if (result.status !== 0) throw new Error('npm publish failed');
await waitUntilVisible(packageJson.name, packageJson.version);

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

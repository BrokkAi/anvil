#!/usr/bin/env node
// Smoke-test the built npm tarballs on the current machine, before any
// registry write. Installs the root and matching platform tarballs into a
// throwaway global prefix with a cold cache, then runs the real native binary
// through the launcher: version output, argument forwarding, exit status,
// signal forwarding, and a one-shot `npm exec` run.
//
// Pre-publish note: the root package's optional platform dependencies do not
// exist in the registry yet, so installs here run with optional dependencies
// omitted and the platform tarball installed explicitly alongside the root.
// The real registry-backed install and npx paths are verified again after
// publication.
//
// Usage: node npm/test-npm-packages.mjs [--dist npm/dist]
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { platformPackageFor, tarballBasename, ROOT_PACKAGE, logStep } from './lib/common.mjs';

const NPM_DIR = path.dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const args = { dist: path.join(NPM_DIR, 'dist') };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--dist') {
      args.dist = path.resolve(argv[i + 1]);
      i += 1;
    } else {
      throw new Error(`unknown argument: ${argv[i]}`);
    }
  }
  return args;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function npmEnv(cacheDir) {
  return {
    ...process.env,
    npm_config_cache: cacheDir,
    // The platform packages are not in the registry until first publication;
    // skip optional-dependency resolution and install the platform tarball
    // explicitly instead.
    npm_config_omit: 'optional',
    npm_config_audit: 'false',
    npm_config_fund: 'false',
    npm_config_update_notifier: 'false',
  };
}

function runNpm(args, env) {
  const result = spawnSync('npm', args, { encoding: 'utf8', env });
  if (result.status !== 0) {
    throw new Error(`npm ${args.join(' ')} failed:\n${result.stdout}\n${result.stderr}`);
  }
  return result;
}

function runLauncher(launcherPath, args) {
  return spawnSync(launcherPath, args, { encoding: 'utf8' });
}

async function testSignalForwarding(launcherPath, nativeBinaryPath) {
  logStep('signal forwarding: SIGTERM on the launcher stops the native process');
  const child = spawn(launcherPath, [], { stdio: ['pipe', 'pipe', 'pipe'] });
  await new Promise((resolve) => setTimeout(resolve, 1500));
  assert(child.exitCode === null, 'anvil server exited before the signal test could run');
  child.kill('SIGTERM');
  const { code, signal } = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('launcher did not exit within 10s of SIGTERM')), 10_000);
    child.on('exit', (exitCode, exitSignal) => {
      clearTimeout(timer);
      resolve({ code: exitCode, signal: exitSignal });
    });
  });
  assert(
    signal === 'SIGTERM' || code !== 0,
    `launcher exited with code=${code} signal=${signal}; expected SIGTERM termination`,
  );
  await new Promise((resolve) => setTimeout(resolve, 500));
  const orphans = spawnSync('pgrep', ['-f', nativeBinaryPath], { encoding: 'utf8' });
  assert(orphans.status !== 0, `native anvil process survived the launcher: pids ${orphans.stdout.trim()}`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const manifest = JSON.parse(fs.readFileSync(path.join(args.dist, 'manifest.json'), 'utf8'));
  const { version } = manifest;

  const target = platformPackageFor(process.platform, process.arch);
  assert(target, `no platform package covers this machine (${process.platform}-${process.arch})`);

  const rootTarball = path.join(args.dist, tarballBasename(ROOT_PACKAGE, version));
  const platformTarball = path.join(args.dist, tarballBasename(target.name, version));
  assert(fs.existsSync(rootTarball), `missing ${rootTarball}; run build-npm-packages.mjs first`);
  assert(fs.existsSync(platformTarball), `missing ${platformTarball}; run build-npm-packages.mjs first`);

  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'anvil-npm-test-'));
  const prefix = path.join(tempDir, 'prefix');
  const cache = path.join(tempDir, 'cache');
  fs.mkdirSync(prefix, { recursive: true });
  const env = npmEnv(cache);

  try {
    logStep(`global install from tarballs into ${prefix}`);
    runNpm(['install', '-g', '--prefix', prefix, platformTarball], env);
    runNpm(['install', '-g', '--prefix', prefix, rootTarball], env);

    const isWindows = process.platform === 'win32';
    const launcher = isWindows ? path.join(prefix, 'anvil.cmd') : path.join(prefix, 'bin', 'anvil');
    assert(fs.existsSync(launcher), `global install did not produce the anvil bin entry at ${launcher}`);
    const nativeBinary = path.join(
      prefix,
      ...(isWindows ? ['node_modules'] : ['lib', 'node_modules']),
      ...target.name.split('/'),
      target.bin,
    );
    assert(fs.existsSync(nativeBinary), `platform package did not install its native binary at ${nativeBinary}`);

    logStep('anvil --version through the global install');
    const versionRun = runLauncher(launcher, ['--version']);
    assert(versionRun.status === 0, `anvil --version exited ${versionRun.status}: ${versionRun.stderr}`);
    assert(
      versionRun.stdout.includes(version),
      `anvil --version output does not mention ${version}: ${versionRun.stdout.trim()}`,
    );

    logStep('argument forwarding and exit status propagation');
    const badFlag = runLauncher(launcher, ['--definitely-not-an-anvil-flag']);
    assert(badFlag.status !== 0, 'a rejected flag should produce a nonzero exit status');
    assert(
      badFlag.stderr.includes('definitely-not-an-anvil-flag'),
      `native stderr did not surface through the launcher: ${badFlag.stderr.trim()}`,
    );

    if (!isWindows) {
      await testSignalForwarding(launcher, nativeBinary);
    }

    logStep('one-shot npm exec from the tarballs');
    const oneShot = spawnSync(
      'npm',
      ['exec', '--yes', `--package=${platformTarball}`, `--package=${rootTarball}`, '--', 'anvil', '--version'],
      { encoding: 'utf8', env, cwd: tempDir },
    );
    assert(oneShot.status === 0, `npm exec one-shot failed:\n${oneShot.stdout}\n${oneShot.stderr}`);
    assert(
      oneShot.stdout.includes(version),
      `npm exec one-shot output does not mention ${version}: ${oneShot.stdout.trim()}`,
    );

    logStep(`smoke tests passed for ${target.name}@${version} on ${process.platform}-${process.arch}`);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

main().catch((error) => {
  process.stderr.write(`test-npm-packages: ${error.message}\n`);
  process.exit(1);
});

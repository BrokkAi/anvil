// Shared definitions for the Anvil npm packaging pipeline.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

export const ROOT_PACKAGE = '@brokkai/anvil';

// One npm platform package per released native target. The root package pins
// each of these as an exact optional dependency at the same product version.
export const PLATFORM_PACKAGES = [
  {
    name: '@brokkai/anvil-darwin-universal',
    rustTarget: 'universal-apple-darwin',
    os: ['darwin'],
    cpu: ['x64', 'arm64'],
    bin: 'anvil',
  },
  {
    name: '@brokkai/anvil-linux-x64',
    rustTarget: 'x86_64-unknown-linux-gnu',
    os: ['linux'],
    cpu: ['x64'],
    libc: ['glibc'],
    bin: 'anvil',
  },
  {
    name: '@brokkai/anvil-linux-arm64',
    rustTarget: 'aarch64-unknown-linux-gnu',
    os: ['linux'],
    cpu: ['arm64'],
    libc: ['glibc'],
    bin: 'anvil',
  },
  {
    name: '@brokkai/anvil-android-arm64',
    rustTarget: 'aarch64-linux-android',
    os: ['android'],
    cpu: ['arm64'],
    bin: 'anvil',
  },
  {
    name: '@brokkai/anvil-win32-x64',
    rustTarget: 'x86_64-pc-windows-msvc',
    os: ['win32'],
    cpu: ['x64'],
    bin: 'anvil.exe',
  },
];

// Files every release bundle carries alongside the binary. Anything not in
// this list (plus the binary and the generated package.json) makes tarball
// validation fail, so credentials or stray development files can never ship.
export const BUNDLE_DOC_FILES = [
  'README.md',
  'LICENSE',
  'GPL-3.0.md',
  'SOURCE.md',
  'THIRD_PARTY_LICENSES.html',
  'SUPPLEMENTAL_THIRD_PARTY_NOTICES.txt',
];

export function platformPackageFor(platform, arch) {
  const key = `${platform}-${arch}`;
  const byKey = {
    'darwin-x64': '@brokkai/anvil-darwin-universal',
    'darwin-arm64': '@brokkai/anvil-darwin-universal',
    'linux-x64': '@brokkai/anvil-linux-x64',
    'linux-arm64': '@brokkai/anvil-linux-arm64',
    'android-arm64': '@brokkai/anvil-android-arm64',
    'win32-x64': '@brokkai/anvil-win32-x64',
  };
  const name = byKey[key];
  if (!name) return undefined;
  return PLATFORM_PACKAGES.find((p) => p.name === name);
}

export function tarballBasename(packageName, version) {
  // npm pack naming: @scope/name -> scope-name-version.tgz
  return `${packageName.replace(/^@/, '').replace('/', '-')}-${version}.tgz`;
}

export function versionFromTag(tag) {
  const match = /^v(\d+\.\d+\.\d+)$/.exec(tag);
  if (!match) {
    throw new Error(`release tag must look like vX.Y.Z, got: ${tag}`);
  }
  return match[1];
}

export function sha256Of(filePath) {
  return createHash('sha256').update(readFileSync(filePath)).digest('hex');
}

export function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    stdio: options.capture ? ['ignore', 'pipe', 'pipe'] : 'inherit',
    encoding: 'utf8',
    ...options.spawn,
  });
  if (result.error) {
    throw new Error(`${command} failed to start: ${result.error.message}`);
  }
  if (result.status !== 0 && !options.allowFailure) {
    const detail = options.capture ? `\n${result.stdout}\n${result.stderr}` : '';
    throw new Error(`${command} ${args.join(' ')} exited with status ${result.status}${detail}`);
  }
  return result;
}

export function logStep(message) {
  process.stdout.write(`\n==> ${message}\n`);
}

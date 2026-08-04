#!/usr/bin/env node
// Build the six @brokkai npm package tarballs for one released Anvil version.
//
// The native payloads are never rebuilt here: every platform package is
// assembled from the checksum-verified GitHub release zip for the given tag.
// Nothing is published; the output is a directory of validated tarballs plus
// a manifest that the publish script consumes.
//
// Usage:
//   node npm/build-npm-packages.mjs --tag v0.24.3
//   node npm/build-npm-packages.mjs --tag v0.24.3 --assets-dir path/to/downloaded/assets
//
// Options:
//   --tag <vX.Y.Z>     Existing GitHub release tag (required).
//   --repo <owner/name> GitHub repository. Defaults to BrokkAi/anvil.
//   --assets-dir <dir> Use pre-downloaded release assets (zips + .sha256)
//                      instead of downloading them.
//   --out <dir>        Output directory. Defaults to npm/dist.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import {
  BUNDLE_DOC_FILES,
  PLATFORM_PACKAGES,
  ROOT_PACKAGE,
  logStep,
  run,
  sha256Of,
  tarballBasename,
  versionFromTag,
} from './lib/common.mjs';

const NPM_DIR = path.dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const args = { repo: 'BrokkAi/anvil', out: path.join(NPM_DIR, 'dist') };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    switch (flag) {
      case '--tag':
        args.tag = value;
        i += 1;
        break;
      case '--repo':
        args.repo = value;
        i += 1;
        break;
      case '--assets-dir':
        args.assetsDir = path.resolve(value);
        i += 1;
        break;
      case '--out':
        args.out = path.resolve(value);
        i += 1;
        break;
      default:
        throw new Error(`unknown argument: ${flag}`);
    }
  }
  if (!args.tag) throw new Error('missing required --tag <vX.Y.Z>');
  return args;
}

async function downloadAsset(repo, tag, assetName, destination) {
  const url = `https://github.com/${repo}/releases/download/${tag}/${assetName}`;
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok) {
    throw new Error(`download failed (${response.status} ${response.statusText}): ${url}`);
  }
  fs.writeFileSync(destination, Buffer.from(await response.arrayBuffer()));
}

function verifyChecksum(zipPath, sidecarPath, assetName) {
  const sidecar = fs.readFileSync(sidecarPath, 'utf8').trim();
  const expected = sidecar.split(/\s+/)[0].toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(expected)) {
    throw new Error(`invalid checksum sidecar for ${assetName}: ${sidecar}`);
  }
  const actual = sha256Of(zipPath);
  if (actual !== expected) {
    throw new Error(`checksum mismatch for ${assetName}: expected ${expected}, got ${actual}`);
  }
  return expected;
}

function extractBundle(zipPath, extractDir, tag, target) {
  fs.mkdirSync(extractDir, { recursive: true });
  run('unzip', ['-q', zipPath, '-d', extractDir]);
  const bundleDir = path.join(extractDir, `brokk-anvil-${tag}-${target.rustTarget}`);
  if (!fs.existsSync(bundleDir)) {
    throw new Error(`release zip for ${target.rustTarget} does not contain expected directory ${path.basename(bundleDir)}`);
  }

  const expected = new Set([...BUNDLE_DOC_FILES, target.bin]);
  const actual = fs.readdirSync(bundleDir).sort();
  const unexpected = actual.filter((entry) => !expected.has(entry));
  const missing = [...expected].filter((entry) => !actual.includes(entry)).sort();
  if (unexpected.length > 0) {
    throw new Error(`release bundle for ${target.rustTarget} contains unexpected entries: ${unexpected.join(', ')}`);
  }
  if (missing.length > 0) {
    throw new Error(`release bundle for ${target.rustTarget} is missing entries: ${missing.join(', ')}`);
  }

  const binaryPath = path.join(bundleDir, target.bin);
  const binarySize = fs.statSync(binaryPath).size;
  if (binarySize < 1024 * 1024) {
    throw new Error(`native binary for ${target.rustTarget} is implausibly small (${binarySize} bytes)`);
  }
  return bundleDir;
}

function basePackageJson(version) {
  return {
    version,
    license: 'LGPL-3.0-only',
    repository: { type: 'git', url: 'git+https://github.com/BrokkAi/anvil.git' },
    homepage: 'https://anvil.brokk.ai/',
    bugs: 'https://github.com/BrokkAi/anvil/issues',
  };
}

function stagePlatformPackage(target, version, bundleDir, stagingRoot) {
  const stageDir = path.join(stagingRoot, target.name.replace('@', '').replace('/', '-'));
  fs.mkdirSync(stageDir, { recursive: true });
  for (const entry of [...BUNDLE_DOC_FILES, target.bin]) {
    fs.copyFileSync(path.join(bundleDir, entry), path.join(stageDir, entry));
  }
  fs.chmodSync(path.join(stageDir, target.bin), 0o755);

  const packageJson = {
    name: target.name,
    ...basePackageJson(version),
    description: `Native Anvil ${target.rustTarget} binary for the @brokkai/anvil npm package`,
    os: target.os,
    cpu: target.cpu,
    ...(target.libc ? { libc: target.libc } : {}),
    // Ask Yarn PnP to keep this package extracted on disk so the native
    // binary can be executed by absolute path.
    preferUnplugged: true,
    files: [...BUNDLE_DOC_FILES, target.bin],
  };
  fs.writeFileSync(path.join(stageDir, 'package.json'), `${JSON.stringify(packageJson, null, 2)}\n`);
  return stageDir;
}

function stageRootPackage(version, anyBundleDir, stagingRoot) {
  const stageDir = path.join(stagingRoot, 'brokkai-anvil');
  fs.mkdirSync(path.join(stageDir, 'bin'), { recursive: true });
  fs.copyFileSync(path.join(NPM_DIR, 'launcher', 'anvil.js'), path.join(stageDir, 'bin', 'anvil.js'));
  fs.chmodSync(path.join(stageDir, 'bin', 'anvil.js'), 0o755);
  fs.copyFileSync(path.join(NPM_DIR, 'launcher', 'README.md'), path.join(stageDir, 'README.md'));
  fs.copyFileSync(path.join(anyBundleDir, 'LICENSE'), path.join(stageDir, 'LICENSE'));

  const optionalDependencies = {};
  for (const target of PLATFORM_PACKAGES) {
    optionalDependencies[target.name] = version;
  }
  const packageJson = {
    name: ROOT_PACKAGE,
    ...basePackageJson(version),
    description: 'Anvil: Rust ACP server with first-run setup for Codex, Ollama, and OpenRouter',
    keywords: ['anvil', 'acp', 'agent-client-protocol', 'agent', 'cli', 'brokk'],
    bin: { anvil: 'bin/anvil.js' },
    files: ['bin/anvil.js'],
    engines: { node: '>=18' },
    optionalDependencies,
  };
  fs.writeFileSync(path.join(stageDir, 'package.json'), `${JSON.stringify(packageJson, null, 2)}\n`);
  return stageDir;
}

function packTarball(stageDir, outDir) {
  const output = execFileSync('npm', ['pack', '--pack-destination', outDir, '--silent'], {
    cwd: stageDir,
    encoding: 'utf8',
  }).trim();
  const tarball = path.join(outDir, output.split('\n').pop().trim());
  if (!fs.existsSync(tarball)) {
    throw new Error(`npm pack did not produce expected tarball: ${tarball}`);
  }
  return tarball;
}

function tarballEntries(tarballPath) {
  const listing = execFileSync('tar', ['-tvzf', tarballPath], { encoding: 'utf8' });
  const entries = new Map();
  for (const line of listing.trim().split('\n')) {
    const fields = line.trim().split(/\s+/);
    const mode = fields[0];
    const name = fields[fields.length - 1];
    entries.set(name, mode);
  }
  return entries;
}

function readTarballPackageJson(tarballPath) {
  const raw = execFileSync('tar', ['-xzOf', tarballPath, 'package/package.json'], { encoding: 'utf8' });
  return JSON.parse(raw);
}

function assertEqualSets(label, actual, expected) {
  const extra = [...actual].filter((entry) => !expected.has(entry)).sort();
  const missing = [...expected].filter((entry) => !actual.has(entry)).sort();
  if (extra.length > 0 || missing.length > 0) {
    throw new Error(
      `${label}: tarball contents differ from allowlist` +
        (extra.length ? `; unexpected: ${extra.join(', ')}` : '') +
        (missing.length ? `; missing: ${missing.join(', ')}` : ''),
    );
  }
}

function validatePlatformTarball(tarballPath, target, version) {
  const entries = tarballEntries(tarballPath);
  const expected = new Set([
    'package/package.json',
    ...BUNDLE_DOC_FILES.map((name) => `package/${name}`),
    `package/${target.bin}`,
  ]);
  assertEqualSets(target.name, new Set(entries.keys()), expected);

  const binaryMode = entries.get(`package/${target.bin}`);
  if (!/^-rwxr-xr-x/.test(binaryMode)) {
    throw new Error(`${target.name}: binary ${target.bin} is not executable in the tarball (mode ${binaryMode})`);
  }

  const manifest = readTarballPackageJson(tarballPath);
  const checks = [
    [manifest.name === target.name, `name is ${manifest.name}`],
    [manifest.version === version, `version is ${manifest.version}`],
    [JSON.stringify(manifest.os) === JSON.stringify(target.os), `os is ${JSON.stringify(manifest.os)}`],
    [JSON.stringify(manifest.cpu) === JSON.stringify(target.cpu), `cpu is ${JSON.stringify(manifest.cpu)}`],
    [manifest.bin === undefined, 'platform packages must not declare bin entries'],
    [manifest.scripts === undefined, 'platform packages must not declare scripts'],
  ];
  for (const [ok, detail] of checks) {
    if (!ok) throw new Error(`${target.name}: ${detail}`);
  }
}

function validateRootTarball(tarballPath, version) {
  const entries = tarballEntries(tarballPath);
  const expected = new Set(['package/package.json', 'package/README.md', 'package/LICENSE', 'package/bin/anvil.js']);
  assertEqualSets(ROOT_PACKAGE, new Set(entries.keys()), expected);

  const manifest = readTarballPackageJson(tarballPath);
  if (manifest.name !== ROOT_PACKAGE) throw new Error(`root package name is ${manifest.name}`);
  if (manifest.version !== version) throw new Error(`root package version is ${manifest.version}`);
  if (JSON.stringify(manifest.bin) !== JSON.stringify({ anvil: 'bin/anvil.js' })) {
    throw new Error(`root package bin is ${JSON.stringify(manifest.bin)}`);
  }
  if (manifest.scripts !== undefined) {
    throw new Error('root package must not declare scripts (no install-time hooks)');
  }
  for (const target of PLATFORM_PACKAGES) {
    if (manifest.optionalDependencies?.[target.name] !== version) {
      throw new Error(`root package must pin ${target.name} to exactly ${version}`);
    }
  }
  if (Object.keys(manifest.optionalDependencies).length !== PLATFORM_PACKAGES.length) {
    throw new Error('root package declares unexpected optional dependencies');
  }
  if (manifest.dependencies !== undefined) {
    throw new Error('root package must not declare runtime dependencies');
  }

  const launcher = execFileSync('tar', ['-xzOf', tarballPath, 'package/bin/anvil.js'], { encoding: 'utf8' });
  if (!launcher.startsWith('#!/usr/bin/env node')) {
    throw new Error('root launcher is missing its node shebang');
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const version = versionFromTag(args.tag);

  fs.rmSync(args.out, { recursive: true, force: true });
  const workDir = path.join(args.out, 'work');
  const assetDir = args.assetsDir ?? path.join(workDir, 'assets');
  fs.mkdirSync(assetDir, { recursive: true });

  const manifest = { tag: args.tag, version, repo: args.repo, packages: [] };
  let rootBundleDir;

  for (const target of PLATFORM_PACKAGES) {
    const assetName = `brokk-anvil-${args.tag}-${target.rustTarget}.zip`;
    const zipPath = path.join(assetDir, assetName);
    const sidecarPath = `${zipPath}.sha256`;

    if (!args.assetsDir) {
      logStep(`downloading ${assetName}`);
      await downloadAsset(args.repo, args.tag, assetName, zipPath);
      await downloadAsset(args.repo, args.tag, `${assetName}.sha256`, sidecarPath);
    } else if (!fs.existsSync(zipPath) || !fs.existsSync(sidecarPath)) {
      throw new Error(`--assets-dir is missing ${assetName} or its .sha256 sidecar`);
    }

    logStep(`verifying and staging ${target.name}`);
    const checksum = verifyChecksum(zipPath, sidecarPath, assetName);
    const bundleDir = extractBundle(zipPath, path.join(workDir, 'extract', target.rustTarget), args.tag, target);
    rootBundleDir ??= bundleDir;

    const stageDir = stagePlatformPackage(target, version, bundleDir, path.join(workDir, 'stage'));
    const tarball = packTarball(stageDir, args.out);
    validatePlatformTarball(tarball, target, version);
    manifest.packages.push({
      name: target.name,
      version,
      tarball: path.basename(tarball),
      sha256: sha256Of(tarball),
      sourceAsset: assetName,
      sourceAssetSha256: checksum,
    });
  }

  logStep(`staging ${ROOT_PACKAGE}`);
  const rootStage = stageRootPackage(version, rootBundleDir, path.join(workDir, 'stage'));
  const rootTarball = packTarball(rootStage, args.out);
  validateRootTarball(rootTarball, version);
  manifest.packages.push({
    name: ROOT_PACKAGE,
    version,
    tarball: path.basename(rootTarball),
    sha256: sha256Of(rootTarball),
  });

  fs.rmSync(workDir, { recursive: true, force: true });
  fs.writeFileSync(path.join(args.out, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);

  logStep('built and validated tarballs');
  for (const pkg of manifest.packages) {
    process.stdout.write(`  ${pkg.name}@${pkg.version}  ${pkg.tarball}  sha256=${pkg.sha256}\n`);
  }
  const expected = tarballBasename(ROOT_PACKAGE, version);
  if (path.basename(rootTarball) !== expected) {
    throw new Error(`unexpected root tarball name: ${path.basename(rootTarball)} (expected ${expected})`);
  }
}

main().catch((error) => {
  process.stderr.write(`build-npm-packages: ${error.message}\n`);
  process.exit(1);
});

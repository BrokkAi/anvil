#!/usr/bin/env node
// Publish the built npm tarballs in the only safe order: every platform
// package first, then — only after all five are publicly visible at the exact
// version — the @brokkai/anvil root package. Safe to re-run after a partial
// publication: versions that already exist in the registry are skipped.
//
// Without --yes-publish this is a dry run that prints the plan and performs
// no registry write.
//
// Auth is ambient: npm trusted publishing (OIDC) in CI, or a logged-in npm
// for the one-time bootstrap publication.
//
// Usage: node npm/publish-npm-packages.mjs [--dist npm/dist] [--yes-publish]
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { ROOT_PACKAGE, logStep, sha256Of } from './lib/common.mjs';
import { versionExists, waitUntilVisible } from './lib/registry.mjs';

const NPM_DIR = path.dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const args = { dist: path.join(NPM_DIR, 'dist'), publish: false };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--dist') {
      args.dist = path.resolve(argv[i + 1]);
      i += 1;
    } else if (argv[i] === '--yes-publish') {
      args.publish = true;
    } else {
      throw new Error(`unknown argument: ${argv[i]}`);
    }
  }
  return args;
}

function npm(args, options = {}) {
  const result = spawnSync('npm', args, { encoding: 'utf8', ...options });
  if (result.error) throw new Error(`npm failed to start: ${result.error.message}`);
  return result;
}

function publishTarball(pkg, distDir, doPublish) {
  const tarball = path.join(distDir, pkg.tarball);
  if (!fs.existsSync(tarball)) throw new Error(`missing tarball ${tarball}; run build-npm-packages.mjs first`);
  const actual = sha256Of(tarball);
  if (actual !== pkg.sha256) {
    throw new Error(`${pkg.tarball} does not match the build manifest (sha256 ${actual} != ${pkg.sha256})`);
  }

  if (versionExists(pkg.name, pkg.version)) {
    process.stdout.write(`  ${pkg.name}@${pkg.version} already published; skipping\n`);
    return;
  }
  if (!doPublish) {
    process.stdout.write(`  would publish ${pkg.name}@${pkg.version} from ${pkg.tarball}\n`);
    return;
  }
  const result = npm(['publish', tarball, '--access', 'public'], { stdio: 'inherit', encoding: undefined });
  if (result.status !== 0) throw new Error(`npm publish ${pkg.name}@${pkg.version} failed`);
}

function verifyPublicInstall(version) {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'anvil-npm-verify-'));
  const prefix = path.join(tempDir, 'prefix');
  fs.mkdirSync(prefix, { recursive: true });
  const env = {
    ...process.env,
    npm_config_cache: path.join(tempDir, 'cache'),
    npm_config_audit: 'false',
    npm_config_fund: 'false',
    npm_config_update_notifier: 'false',
  };
  try {
    logStep(`verifying public install of ${ROOT_PACKAGE}@${version} from a clean environment`);
    const install = npm(['install', '-g', '--prefix', prefix, `${ROOT_PACKAGE}@${version}`], { env });
    if (install.status !== 0) throw new Error(`public global install failed:\n${install.stdout}\n${install.stderr}`);
    const launcher =
      process.platform === 'win32' ? path.join(prefix, 'anvil.cmd') : path.join(prefix, 'bin', 'anvil');
    const versionRun = spawnSync(launcher, ['--version'], { encoding: 'utf8' });
    if (versionRun.status !== 0 || !versionRun.stdout.includes(version)) {
      throw new Error(`public install smoke test failed: ${versionRun.stdout} ${versionRun.stderr}`);
    }

    logStep(`verifying one-shot npx -y ${ROOT_PACKAGE}@${version} from a cold cache`);
    const npx = spawnSync('npx', ['-y', `${ROOT_PACKAGE}@${version}`, '--version'], {
      encoding: 'utf8',
      env: { ...env, npm_config_cache: path.join(tempDir, 'npx-cache') },
      cwd: tempDir,
    });
    if (npx.status !== 0 || !npx.stdout.includes(version)) {
      throw new Error(`npx one-shot verification failed: ${npx.stdout} ${npx.stderr}`);
    }
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const manifest = JSON.parse(fs.readFileSync(path.join(args.dist, 'manifest.json'), 'utf8'));
  const platformPackages = manifest.packages.filter((pkg) => pkg.name !== ROOT_PACKAGE);
  const rootPackage = manifest.packages.find((pkg) => pkg.name === ROOT_PACKAGE);
  if (!rootPackage || platformPackages.length === 0) {
    throw new Error('manifest.json does not describe the expected package set');
  }

  logStep(args.publish ? `publishing ${manifest.tag} packages` : `dry run for ${manifest.tag} (no registry writes)`);
  for (const pkg of platformPackages) {
    publishTarball(pkg, args.dist, args.publish);
  }

  if (!args.publish) {
    process.stdout.write(`  would then wait for visibility and publish ${rootPackage.name}@${rootPackage.version}\n`);
    return;
  }

  logStep('waiting for all platform packages to be publicly visible');
  for (const pkg of platformPackages) {
    await waitUntilVisible(pkg.name, pkg.version);
  }

  logStep(`publishing ${ROOT_PACKAGE} (all platform packages verified visible)`);
  publishTarball(rootPackage, args.dist, true);
  await waitUntilVisible(rootPackage.name, rootPackage.version);

  verifyPublicInstall(rootPackage.version);
  logStep('publication complete and verified');
}

main().catch((error) => {
  process.stderr.write(`publish-npm-packages: ${error.message}\n`);
  process.exit(1);
});

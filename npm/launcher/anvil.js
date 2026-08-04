#!/usr/bin/env node
// Launcher for the @brokkai/anvil npm package.
//
// The native `anvil` binary ships in per-platform packages that the root
// package pins as exact optional dependencies. This script picks the package
// matching the current platform, executes the binary by absolute path, and
// forwards arguments, stdio, signals, and the exit status unchanged.
'use strict';

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const PLATFORM_PACKAGES = {
  'darwin-arm64': { name: '@brokkai/anvil-darwin-universal', bin: 'anvil' },
  'darwin-x64': { name: '@brokkai/anvil-darwin-universal', bin: 'anvil' },
  'linux-arm64': { name: '@brokkai/anvil-linux-arm64', bin: 'anvil' },
  'linux-x64': { name: '@brokkai/anvil-linux-x64', bin: 'anvil' },
  'android-arm64': { name: '@brokkai/anvil-android-arm64', bin: 'anvil' },
  'win32-x64': { name: '@brokkai/anvil-win32-x64', bin: 'anvil.exe' },
};

function fail(message) {
  process.stderr.write(`anvil (npm launcher): ${message}\n`);
  process.exit(1);
}

function resolveBinary() {
  const key = `${process.platform}-${process.arch}`;
  const entry = PLATFORM_PACKAGES[key];
  if (!entry) {
    fail(
      `no prebuilt Anvil binary is published for ${key}. ` +
        `Supported platforms: ${Object.keys(PLATFORM_PACKAGES).join(', ')}. ` +
        'Install from source instead: https://anvil.brokk.ai/install/',
    );
  }

  let packageJsonPath;
  try {
    packageJsonPath = require.resolve(`${entry.name}/package.json`);
  } catch {
    fail(
      `the platform package ${entry.name} is not installed. ` +
        'It is an optional dependency of @brokkai/anvil; reinstall without ' +
        '--no-optional / --omit=optional so npm can install it.',
    );
  }

  const binaryPath = path.join(path.dirname(packageJsonPath), entry.bin);
  if (!fs.existsSync(binaryPath)) {
    fail(`the platform package ${entry.name} is missing its binary at ${binaryPath}; reinstall @brokkai/anvil.`);
  }
  return binaryPath;
}

function run() {
  const binaryPath = resolveBinary();
  const child = spawn(binaryPath, process.argv.slice(2), { stdio: 'inherit' });

  const forwarded = ['SIGINT', 'SIGTERM', 'SIGHUP'];
  for (const signal of forwarded) {
    process.on(signal, () => {
      child.kill(signal);
    });
  }

  child.on('error', (error) => {
    fail(`failed to launch ${binaryPath}: ${error.message}`);
  });

  child.on('exit', (code, signal) => {
    if (signal) {
      // Re-raise the fatal signal so callers observe the same termination
      // status the native binary had. Reset our forwarding handler first so
      // the re-raised signal terminates this process instead of re-entering it.
      process.removeAllListeners(signal);
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code === null ? 1 : code);
  });
}

run();

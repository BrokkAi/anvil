import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { nativeBinaryName, platformPackageFor } from "../lib/platform.mjs";
import { createPlatformStage, createRootStage } from "../scripts/build-packages.mjs";

const NPM_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

test("platform selection matches every Anvil release target", () => {
  assert.equal(platformPackageFor("darwin", "arm64").packageName, "@brokkai/anvil-darwin-universal");
  assert.equal(platformPackageFor("darwin", "x64").target, "universal-apple-darwin");
  assert.equal(platformPackageFor("linux", "x64").target, "x86_64-unknown-linux-gnu");
  assert.equal(platformPackageFor("linux", "arm64").target, "aarch64-unknown-linux-gnu");
  assert.equal(platformPackageFor("android", "arm64").target, "aarch64-linux-android");
  assert.equal(platformPackageFor("win32", "x64").target, "x86_64-pc-windows-msvc");
  assert.throws(() => platformPackageFor("win32", "arm64"), /does not support/);
  assert.equal(nativeBinaryName("win32"), "anvil.exe");
});

test("root package exposes anvil and pins distinct stable platform packages", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "anvil-npm-root-test-"));
  try {
    const packageJson = createRootStage({ stageDir: root, version: "0.24.2" });
    assert.equal(packageJson.name, "@brokkai/anvil");
    assert.deepEqual(packageJson.bin, { anvil: "bin/anvil.js" });
    assert.equal(packageJson.optionalDependencies["@brokkai/anvil-linux-x64"], "0.24.2");
    assert.equal(Object.keys(packageJson.optionalDependencies).length, 5);
  } finally {
    fs.rmSync(root, { force: true, recursive: true });
  }
});

test("platform package keeps the complete release bundle", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "anvil-npm-platform-test-"));
  const bundle = path.join(root, "bundle");
  const stage = path.join(root, "stage");
  fs.mkdirSync(bundle);
  fs.writeFileSync(path.join(bundle, "anvil"), "binary");
  fs.writeFileSync(path.join(bundle, "SOURCE.md"), "source");

  try {
    const packageJson = createPlatformStage({
      bundleDir: bundle,
      platformKey: "linux-x64",
      stageDir: stage,
      version: "0.24.2",
    });
    assert.equal(packageJson.name, "@brokkai/anvil-linux-x64");
    assert.equal(packageJson.version, "0.24.2");
    assert.equal(
      fs.readFileSync(path.join(stage, "vendor", "x86_64-unknown-linux-gnu", "anvil"), "utf8"),
      "binary",
    );
    assert.ok(fs.existsSync(path.join(stage, "vendor", "x86_64-unknown-linux-gnu", "SOURCE.md")));
  } finally {
    fs.rmSync(root, { force: true, recursive: true });
  }
});

test("platform package rejects a release bundle without Anvil", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "anvil-npm-missing-test-"));
  const bundle = path.join(root, "bundle");
  fs.mkdirSync(bundle);
  try {
    assert.throws(
      () => createPlatformStage({
        bundleDir: bundle,
        platformKey: "linux-x64",
        stageDir: path.join(root, "stage"),
        version: "0.24.2",
      }),
      /missing anvil/,
    );
  } finally {
    fs.rmSync(root, { force: true, recursive: true });
  }
});

test("launcher forwards arguments and the native exit status", { skip: process.platform === "win32" }, () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "anvil-npm-launcher-test-"));
  const packageRoot = path.join(root, "package");
  const selected = platformPackageFor(process.platform, process.arch);
  const nativeRoot = path.join(packageRoot, "node_modules", ...selected.packageName.split("/"));
  const vendor = path.join(nativeRoot, "vendor", selected.target);
  fs.mkdirSync(path.join(packageRoot, "bin"), { recursive: true });
  fs.mkdirSync(path.join(packageRoot, "lib"), { recursive: true });
  fs.mkdirSync(vendor, { recursive: true });
  fs.copyFileSync(path.join(NPM_DIR, "bin", "anvil.js"), path.join(packageRoot, "bin", "anvil.js"));
  fs.copyFileSync(path.join(NPM_DIR, "lib", "platform.mjs"), path.join(packageRoot, "lib", "platform.mjs"));
  fs.writeFileSync(path.join(nativeRoot, "package.json"), JSON.stringify({ name: selected.packageName, version: "0.24.2" }));
  const fakeAnvil = path.join(vendor, "anvil");
  fs.writeFileSync(fakeAnvil, '#!/bin/sh\nprintf "%s\\n" "$1"\nexit 23\n');
  fs.chmodSync(fakeAnvil, 0o755);

  try {
    const result = spawnSync(process.execPath, [path.join(packageRoot, "bin", "anvil.js"), "hello"], { encoding: "utf8" });
    assert.equal(result.status, 23, result.stderr);
    assert.equal(result.stdout, "hello\n");
  } finally {
    fs.rmSync(root, { force: true, recursive: true });
  }
});

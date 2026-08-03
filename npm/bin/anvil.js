#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import path from "node:path";
import process from "node:process";

import { nativeBinaryName, platformPackageFor } from "../lib/platform.mjs";

const require = createRequire(import.meta.url);

function resolveNativeBinary() {
  const selected = platformPackageFor(process.platform, process.arch);
  let packageJson;
  try {
    packageJson = require.resolve(`${selected.packageName}/package.json`);
  } catch (error) {
    throw new Error(
      `The native ${selected.packageName} payload is missing. ` +
        "Reinstall @brokkai/anvil with optional dependencies enabled.",
      { cause: error },
    );
  }

  return path.join(
    path.dirname(packageJson),
    "vendor",
    selected.target,
    nativeBinaryName(process.platform),
  );
}

let binary;
try {
  binary = resolveNativeBinary();
} catch (error) {
  console.error(`Error: ${error.message}`);
  process.exit(1);
}

const child = spawn(binary, process.argv.slice(2), { stdio: "inherit" });

for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(signal, () => {
    if (!child.killed) {
      child.kill(signal);
    }
  });
}

child.on("error", (error) => {
  console.error(`Error: failed to launch ${binary}: ${error.message}`);
  process.exit(1);
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 1);
});

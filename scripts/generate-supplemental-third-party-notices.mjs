#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);
const outputPath = path.resolve(
  repositoryRoot,
  process.argv[2] ?? "licenses/SUPPLEMENTAL_THIRD_PARTY_NOTICES.txt",
);

// cargo-about handles the license files declared by Cargo packages. This
// inventory catches standalone notices and native payloads that can otherwise
// appear without anyone deciding whether an additional artifact notice is due.
const auditedStandaloneNotices = new Set(["cfg_aliases/NOTICES.md"]);
const auditedLinksPackages = new Set([
  "aws-lc-rs",
  "aws-lc-sys",
  "libsqlite3-sys",
  "prettyplease",
  "rayon-core",
  "ring",
  "wasm-bindgen-shared",
  "zstd-sys",
]);

function cargoMetadata() {
  return JSON.parse(
    execFileSync("cargo", ["metadata", "--locked", "--format-version", "1"], {
      cwd: repositoryRoot,
      encoding: "utf8",
      maxBuffer: 32 * 1024 * 1024,
    }),
  );
}

function resolvedPackageIds(metadata) {
  return new Set(metadata.resolve.nodes.map(({ id }) => id));
}

function resolvedPackage(metadata, name) {
  const resolvedIds = resolvedPackageIds(metadata);
  const matches = metadata.packages.filter(
    (packageInfo) =>
      packageInfo.name === name && resolvedIds.has(packageInfo.id),
  );
  if (matches.length !== 1) {
    throw new Error(
      `expected exactly one resolved ${name} package, found ${matches.length}`,
    );
  }
  return matches[0];
}

function checkNativePackageInventory(metadata) {
  const resolvedIds = resolvedPackageIds(metadata);
  const unknown = metadata.packages
    .filter(
      (packageInfo) =>
        resolvedIds.has(packageInfo.id) &&
        packageInfo.links &&
        !auditedLinksPackages.has(packageInfo.name),
    )
    .map(({ name, version, links }) => `${name}@${version} (links=${links})`)
    .sort();
  if (unknown.length > 0) {
    throw new Error(
      `unaudited native-linking packages in the locked graph:\n${unknown.join("\n")}`,
    );
  }
}

async function checkStandaloneNoticeInventory(metadata) {
  const resolvedIds = resolvedPackageIds(metadata);
  const discovered = [];
  for (const packageInfo of metadata.packages) {
    if (!resolvedIds.has(packageInfo.id)) {
      continue;
    }
    const filenames = await readdir(packageRoot(packageInfo));
    for (const filename of filenames) {
      if (/^NOTICES?(?:\..*)?$/i.test(filename)) {
        discovered.push(`${packageInfo.name}/${filename}`);
      }
    }
  }
  const unknown = discovered
    .filter((notice) => !auditedStandaloneNotices.has(notice))
    .sort();
  if (unknown.length > 0) {
    throw new Error(
      `unaudited standalone notice files in the locked graph:\n${unknown.join("\n")}`,
    );
  }
}

function packageRoot(packageInfo) {
  return path.dirname(packageInfo.manifest_path);
}

function packageUrl(packageInfo) {
  return `https://crates.io/crates/${packageInfo.name}/${encodeURIComponent(packageInfo.version)}`;
}

async function legalFile(metadata, name, relativePath, component, scope) {
  const packageInfo = resolvedPackage(metadata, name);
  const text = (
    await readFile(path.join(packageRoot(packageInfo), relativePath), "utf8")
  ).trimEnd();
  if (!text) {
    throw new Error(`${name}/${relativePath} is empty`);
  }
  return { component, packageInfo, relativePath, scope, text };
}

async function sqliteNotice(metadata) {
  const packageInfo = resolvedPackage(metadata, "libsqlite3-sys");
  const relativePath = "sqlite3/sqlite3.c";
  const source = await readFile(
    path.join(packageRoot(packageInfo), relativePath),
    "utf8",
  );
  const version = source.match(/^#define SQLITE_VERSION\s+"([^"]+)"/m)?.[1];
  const notice = source.match(
    /^\*\* The author disclaims copyright to this source code\. +In place of\n\*\* a legal notice, here is a blessing:\n\*\*\n\*\*    May you do good and not evil\.\n\*\*    May you find forgiveness for yourself and forgive others\.\n\*\*    May you share freely, never taking more than you give\./m,
  )?.[0];
  if (!version || !notice) {
    throw new Error("could not find SQLite version and public-domain notice");
  }
  const text = notice
    .split("\n")
    .map((line) => line.replace(/^\*\* ?/, ""))
    .join("\n");
  return {
    component: `SQLite ${version}`,
    packageInfo,
    relativePath: `${relativePath} (public-domain notice)`,
    scope: "compiled from the bundled SQLite amalgamation",
    text,
  };
}

function render(sections) {
  const lines = [
    "ANVIL SUPPLEMENTAL THIRD-PARTY NOTICES",
    "",
    "This file supplements THIRD_PARTY_LICENSES.html. Cargo package metadata",
    "does not enumerate standalone NOTICE files or every license embedded in",
    "native source trees compiled by Rust wrapper crates.",
    "",
    "The sections below are reproduced from exact packages resolved by",
    "Cargo.lock for Anvil's default release feature set. The generated Rust",
    "report also covers the wasm32-wasip2 parser guest embedded in the binary.",
  ];

  for (const section of sections) {
    const { packageInfo } = section;
    lines.push(
      "",
      "=".repeat(80),
      section.component,
      "=".repeat(80),
      "",
      `Rust package: ${packageInfo.name}@${packageInfo.version}`,
      `Package source: ${packageUrl(packageInfo)}`,
      `Source notice: ${section.relativePath}`,
      `Inclusion: ${section.scope}`,
      "",
      section.text,
    );
  }
  return `${lines.join("\n")}\n`;
}

async function main() {
  const metadata = cargoMetadata();
  checkNativePackageInventory(metadata);
  await checkStandaloneNoticeInventory(metadata);
  const sections = [
    await legalFile(
      metadata,
      "cfg_aliases",
      "NOTICES.md",
      "cfg_aliases third-party attribution",
      "used while compiling target-specific dependency configuration",
    ),
    await sqliteNotice(metadata),
    await legalFile(
      metadata,
      "zstd-sys",
      "zstd/LICENSE",
      "Zstandard",
      "compiled from bundled Zstandard source for ZIP compression support",
    ),
  ];
  await writeFile(outputPath, render(sections), "utf8");
  process.stdout.write(`Wrote supplemental notices to ${outputPath}\n`);
}

await main();

#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const packageDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repositoryDir = path.resolve(packageDir, '..', '..');
const packageJson = JSON.parse(await readFile(path.join(packageDir, 'package.json'), 'utf8'));
const cargoToml = await readFile(path.join(repositoryDir, 'Cargo.toml'), 'utf8');
const cargoVersion = cargoToml.match(/^version = "([^"]+)"/m)?.[1];

if (!cargoVersion) {
  throw new Error('could not read the brokk-anvil version from Cargo.toml');
}

if (packageJson.version !== cargoVersion) {
  throw new Error(
    `version mismatch: ${packageJson.name}=${packageJson.version}, brokk-anvil=${cargoVersion}`,
  );
}

process.stdout.write(`TypeScript SDK version matches Cargo.toml: ${cargoVersion}\n`);

#!/usr/bin/env node
import { copyFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const packageDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repositoryDir = path.resolve(packageDir, '..', '..');
const licenseDir = path.join(packageDir, 'dist', 'licenses');
await mkdir(licenseDir, { recursive: true });
await copyFile(path.join(repositoryDir, 'LICENSE'), path.join(licenseDir, 'LGPL-3.0.txt'));
await copyFile(
  path.join(repositoryDir, 'licenses', 'GPL-3.0.md'),
  path.join(licenseDir, 'GPL-3.0.md'),
);

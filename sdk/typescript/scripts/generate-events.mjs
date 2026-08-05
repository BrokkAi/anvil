#!/usr/bin/env node
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { compileFromFile } from 'json-schema-to-typescript';

const packageDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repositoryDir = path.resolve(packageDir, '..', '..');
const schemaPath = path.join(repositoryDir, 'openapi', 'anvil.v1.events.schema.json');
const outputPath = path.join(packageDir, 'src', 'generated', 'events.gen.ts');
const generated = await compileFromFile(schemaPath, {
  bannerComment: [
    '// Generated from openapi/anvil.v1.events.schema.json (Anvil Agent API contract 1.0.0).',
    '// Generator: json-schema-to-typescript 15.0.4. Do not edit by hand.',
  ].join('\n'),
  cwd: path.dirname(schemaPath),
  enableConstEnums: false,
  style: {
    singleQuote: true,
    trailingComma: 'all',
  },
  unknownAny: true,
});

await mkdir(path.dirname(outputPath), { recursive: true });
await writeFile(outputPath, generated, 'utf8');

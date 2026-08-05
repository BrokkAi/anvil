#!/usr/bin/env node
import { mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const repositoryDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sdkDir = path.join(repositoryDir, 'sdk');
const toolingDir = path.join(sdkDir, 'typescript');
const require = createRequire(path.join(toolingDir, 'package.json'));
const { parse } = require('yaml');
const generatedContractDir = path.join(sdkDir, '.generated');
const generatedContractPath = path.join(generatedContractDir, 'anvil.v1.sdk.json');
const rustSdkDir = path.join(sdkDir, 'rust');
const pythonSdkDir = path.join(sdkDir, 'python');

function run(command, args, cwd = repositoryDir) {
  const result = spawnSync(command, args, { cwd, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

async function normalizeGeneratedTextTree(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  await Promise.all(
    entries.map(async (entry) => {
      const entryPath = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        await normalizeGeneratedTextTree(entryPath);
        return;
      }
      const original = await readFile(entryPath, 'utf8');
      const normalized = `${original.replace(/[ \t]+$/gm, '').replace(/\n+$/, '')}\n`;
      if (normalized !== original) await writeFile(entryPath, normalized, 'utf8');
    }),
  );
}

function eventSchemaName(name) {
  return `Event${name[0].toUpperCase()}${name.slice(1)}`;
}

function rewriteEventRefs(value) {
  if (Array.isArray(value)) return value.map(rewriteEventRefs);
  if (value === null || typeof value !== 'object') return value;

  return Object.fromEntries(
    Object.entries(value).map(([key, child]) => {
      if (key === '$ref' && typeof child === 'string' && child.startsWith('#/$defs/')) {
        return [key, `#/components/schemas/${eventSchemaName(child.slice('#/$defs/'.length))}`];
      }
      return [key, rewriteEventRefs(child)];
    }),
  );
}

async function prepareSdkContract() {
  const openapiPath = path.join(repositoryDir, 'openapi', 'anvil.v1.yaml');
  const eventsPath = path.join(repositoryDir, 'openapi', 'anvil.v1.events.schema.json');
  const openapi = parse(await readFile(openapiPath, 'utf8'));
  const events = JSON.parse(await readFile(eventsPath, 'utf8'));

  for (const [name, schema] of Object.entries(events.$defs)) {
    openapi.components.schemas[eventSchemaName(name)] = rewriteEventRefs(schema);
  }

  const { $schema: _schema, $id: _id, $defs: _defs, ...eventRoot } = events;
  openapi.components.schemas.AnvilRunEvent = {
    ...rewriteEventRefs(eventRoot),
    description: events.description,
  };
  openapi.paths['/v1/runs/{run_id}/events'].get.responses['200'].content[
    'text/event-stream'
  ].schema = { $ref: '#/components/schemas/AnvilRunEvent' };

  await mkdir(generatedContractDir, { recursive: true });
  await writeFile(generatedContractPath, `${JSON.stringify(openapi, null, 2)}\n`, 'utf8');
}

await prepareSdkContract();
run('npm', ['exec', '--', 'openapi-ts'], toolingDir);

const cargoManifest = await readFile(path.join(repositoryDir, 'Cargo.toml'), 'utf8');
const sdkVersion = cargoManifest.match(/^version = "([^"]+)"/m)?.[1];
if (!sdkVersion) throw new Error('could not read package version from Cargo.toml');

await rm(rustSdkDir, { recursive: true, force: true });
run(
  'npm',
  [
    'exec',
    '--',
    'openapi-generator-cli',
    'generate',
    '-g',
    'rust',
    '-i',
    generatedContractPath,
    '-o',
    rustSdkDir,
    '--additional-properties',
    [
      'packageName=brokk-anvil-sdk',
      `packageVersion=${sdkVersion}`,
      'library=reqwest',
      'reqwestDefaultFeatures=rustls',
      'supportAsync=true',
      'useSingleRequestParameter=true',
      'useSerdePathToError=true',
      'preferUnsignedInt=true',
      'repositoryUrl=https://github.com/BrokkAi/anvil',
      'documentationUrl=https://docs.rs/brokk-anvil-sdk',
      'homePageUrl=https://github.com/BrokkAi/anvil',
    ].join(','),
    '--global-property',
    'apiDocs=false,modelDocs=false,apiTests=false,modelTests=false',
  ],
  toolingDir,
);
await Promise.all(
  ['.travis.yml', 'git_push.sh'].map((name) =>
    rm(path.join(rustSdkDir, name), { recursive: true, force: true }),
  ),
);
run('cargo', ['fmt', '-p', 'brokk-anvil-sdk'], repositoryDir);
await normalizeGeneratedTextTree(rustSdkDir);

await rm(pythonSdkDir, { recursive: true, force: true });
run(
  'npm',
  [
    'exec',
    '--',
    'openapi-generator-cli',
    'generate',
    '-g',
    'python',
    '-i',
    generatedContractPath,
    '-o',
    pythonSdkDir,
    '--additional-properties',
    [
      'packageName=brokk_anvil_sdk',
      'projectName=brokk-anvil-sdk',
      `packageVersion=${sdkVersion}`,
      'library=asyncio',
      'disallowAdditionalPropertiesIfNotPresent=false',
    ].join(','),
    '--git-user-id',
    'BrokkAi',
    '--git-repo-id',
    'anvil',
    '--global-property',
    'apiDocs=false,modelDocs=false,apiTests=false,modelTests=false',
  ],
  toolingDir,
);
await Promise.all(
  [
    '.github',
    '.gitlab-ci.yml',
    '.travis.yml',
    'git_push.sh',
    'test-requirements.txt',
    'tox.ini',
  ].map((name) => rm(path.join(pythonSdkDir, name), { recursive: true, force: true })),
);
await normalizeGeneratedTextTree(pythonSdkDir);

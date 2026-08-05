// Registry visibility checks shared by the npm publish scripts.
//
// npmjs serves packuments with `cache-control: public, max-age=300`, so the
// existence check every publish script runs *before* publishing leaves npm (and
// the CDN edge in front of it) answering with pre-publish data for up to five
// minutes afterwards. A freshly published version therefore looks like an E404
// long after the publish itself succeeded — the failure mode that made a
// completed 0.25.0 SDK publication report as a failed job.
//
// Two things keep that from happening: `--prefer-online` forces npm to
// revalidate instead of trusting its local copy, and the wait window is longer
// than the registry's own cache lifetime so an edge still serving a cached
// packument cannot outlast the poll.
import { spawnSync } from 'node:child_process';

export const VISIBILITY_ATTEMPTS = 40;
export const VISIBILITY_DELAY_MS = 15_000;

export function versionExists(name, version) {
  const result = spawnSync('npm', ['view', `${name}@${version}`, 'version', '--json', '--prefer-online'], {
    encoding: 'utf8',
  });
  if (result.error) throw new Error(`npm failed to start: ${result.error.message}`);
  if (result.status === 0) {
    const parsed = JSON.parse(result.stdout.trim() || 'null');
    return parsed === version || (Array.isArray(parsed) && parsed.includes(version));
  }
  // npm reports both "no such package" and "no such version of a known
  // package" as E404; either means the version is not visible yet.
  if (/E404/.test(result.stderr)) return false;
  throw new Error(`npm view ${name}@${version} failed:\n${result.stderr}`);
}

export async function waitUntilVisible(name, version) {
  for (let attempt = 1; attempt <= VISIBILITY_ATTEMPTS; attempt += 1) {
    if (versionExists(name, version)) return;
    process.stdout.write(`  ${name}@${version} not visible yet (attempt ${attempt}/${VISIBILITY_ATTEMPTS})\n`);
    await new Promise((resolve) => setTimeout(resolve, VISIBILITY_DELAY_MS));
  }
  throw new Error(
    `${name}@${version} did not become visible in the registry after ` +
      `${Math.round((VISIBILITY_ATTEMPTS * VISIBILITY_DELAY_MS) / 60_000)} minutes; ` +
      'the publish itself may still have succeeded — check https://www.npmjs.com/package/' +
      `${name}?activeTab=versions before re-running`,
  );
}

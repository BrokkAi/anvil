import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const docsRoot = fileURLToPath(new URL('..', import.meta.url));
const fixture = readFileSync(`${docsRoot}/fixtures/ten-minute-evaluation/src/lib.rs`, 'utf8');
const guide = readFileSync(`${docsRoot}/src/content/docs/evaluate-anvil.md`, 'utf8');

const expectedFixture = `pub fn greeting(name: &str) -> String {
    format!("Hello, {name}!")
}

pub fn welcome(name: &str) -> String {
    greeting(name)
}
`;

if (fixture !== expectedFixture) {
  throw new Error('The ten-minute evaluation fixture changed; update its guide and consistency check together.');
}

for (const required of ['src/lib.rs:1', 'src/lib.rs:6', 'Hello, {name}!', 'Welcome, {name}!']) {
  if (!guide.includes(required)) {
    throw new Error(`The ten-minute evaluation guide is missing the checked expectation: ${required}`);
  }
}

console.log('Checked the ten-minute evaluation fixture and documented line expectations.');

import { AnvilClient } from '@brokkai/anvil-sdk';

const client = new AnvilClient({ token: process.env.ANVIL_TOKEN });
const session = await client.createSession({
  cwd: process.cwd(),
  permission_mode: 'acceptEdits',
});

try {
  const run = await session.run('Summarize this repository.');
  console.log((await run.wait()).result_text);
} finally {
  await session.delete();
}

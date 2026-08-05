import { AnvilClient } from '@brokkai/anvil-sdk';

const client = new AnvilClient({ token: process.env.ANVIL_TOKEN });
const session = await client.createSession({ cwd: process.cwd() });

try {
  const run = await session.run({
    prompt: 'Is this repository ready to release?',
    structured_output: {
      schema_name: 'release_readiness',
      schema: {
        type: 'object',
        required: ['ready', 'reason'],
        properties: {
          ready: { type: 'boolean' },
          reason: { type: 'string' },
        },
      },
    },
  });

  console.log((await run.wait()).structured_output);
} finally {
  await session.delete();
}

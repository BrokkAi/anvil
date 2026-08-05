import { AnvilClient } from '@brokkai/anvil-sdk';

const client = new AnvilClient({ token: process.env.ANVIL_TOKEN });
const session = await client.createSession({ cwd: process.cwd() });
const run = await session.run('Inspect the project and make a plan.');

for await (const event of run.events()) {
  if (event.type === 'message.delta') process.stdout.write(event.text);
  if (event.type === 'plan.updated') console.error(event.plan);
  if (event.type.startsWith('tool_call.')) console.error(event.type, event.tool_name);
}

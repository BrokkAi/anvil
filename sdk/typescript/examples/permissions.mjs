import { AnvilClient } from '@brokkai/anvil-sdk';

const client = new AnvilClient({ token: process.env.ANVIL_TOKEN });
const session = await client.createSession({ cwd: process.cwd(), permission_mode: 'default' });
const run = await session.run('Create hello.txt.');

const result = await run.wait({
  onPermission(permission) {
    const allow = permission.options.find((option) => option.kind === 'allow_once');
    return allow?.id ?? { cancel: true };
  },
});
console.log(result.result_text);

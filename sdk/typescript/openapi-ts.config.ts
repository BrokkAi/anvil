import { defineConfig } from '@hey-api/openapi-ts';

export default defineConfig({
  input: '../../openapi/anvil.v1.yaml',
  output: {
    path: 'src/generated/openapi',
    clean: true,
    header: [
      '// Generated from openapi/anvil.v1.yaml (Anvil Agent API contract 1.0.0).',
      '// Generator: @hey-api/openapi-ts 0.99.0. Do not edit by hand.',
    ],
    module: {
      extension: '.js',
    },
  },
  plugins: [
    {
      name: '@hey-api/client-fetch',
      throwOnError: true,
    },
    '@hey-api/typescript',
    {
      name: '@hey-api/sdk',
      responseStyle: 'data',
    },
  ],
});

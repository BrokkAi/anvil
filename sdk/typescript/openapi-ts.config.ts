import { defineConfig } from '@hey-api/openapi-ts';

export default defineConfig({
  input: '../.generated/anvil.v1.sdk.json',
  output: {
    path: 'src/generated/openapi',
    clean: true,
    header: [
      '// Generated from openapi/anvil.v1.yaml and openapi/anvil.v1.events.schema.json.',
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

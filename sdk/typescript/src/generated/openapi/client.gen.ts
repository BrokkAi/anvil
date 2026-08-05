// Generated from openapi/anvil.v1.yaml (Anvil Agent API contract 1.0.0).
// Generator: @hey-api/openapi-ts 0.99.0. Do not edit by hand.

import { type Client, type ClientOptions, type Config, createClient, createConfig } from './client/index.js';
import type { ClientOptions as ClientOptions2 } from './types.gen.js';

/**
 * The `createClientConfig()` function will be called on client initialization
 * and the returned object will become the client's initial configuration.
 *
 * You may want to initialize your client this way instead of calling
 * `setConfig()`. This is useful for example if you're using Next.js
 * to ensure your client always has the correct values.
 */
export type CreateClientConfig<T extends ClientOptions = ClientOptions2> = (override?: Config<ClientOptions & T>) => Config<Required<ClientOptions> & T>;

export const client: Client = createClient(createConfig<ClientOptions2>({ baseUrl: 'http://127.0.0.1:26845', throwOnError: true }));

import assert from 'node:assert/strict';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { initTheme } from '@earendil-works/pi-coding-agent';

process.env['PI_CODING_AGENT_DIR'] = await mkdtemp(join(tmpdir(), 'pi-oracle-'));
const { default: extension } = await import('../index.ts');
initTheme('dark');

type Registered = { tools: Map<string, any>; commands: Map<string, any>; sent: any[] };

function load(): Registered {
  const registered: Registered = { tools: new Map(), commands: new Map(), sent: [] };
  extension({
    registerTool: (tool: any) => registered.tools.set(tool.name, tool),
    registerCommand: (name: string, command: any) => registered.commands.set(name, command),
    registerMessageRenderer: () => {},
    sendMessage: (message: any) => registered.sent.push(message),
  } as any);
  return registered;
}

function model(provider: string, id: string) {
  return {
    provider,
    id,
    name: id,
    api: 'openai-responses',
    baseUrl: 'https://default.example',
    reasoning: false,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 200_000,
    maxTokens: 32_000,
  };
}

function modelRegistry(models: any[], requests: any[]) {
  return {
    find: (provider: string, id: string) =>
      models.find((item) => item.provider === provider && item.id === id),
    refresh: async () => {},
    getAvailable: () => models,
    getApiKeyAndHeaders: async () => ({
      ok: true,
      apiKey: 'key',
      baseUrl: 'https://enterprise.example',
    }),
    getProvider: () => ({
      streamSimple: (requestModel: any) => {
        requests.push(requestModel);
        return {
          result: async () => ({
            role: 'assistant',
            stopReason: 'stop',
            content: [{ type: 'text', text: 'Looks fine.' }],
            usage: {
              input: 1,
              output: 1,
              cacheRead: 0,
              cacheWrite: 0,
              cost: { total: 0 },
            },
          }),
        };
      },
    }),
  };
}

test('oracle tool sends requests to the base URL resolved by authentication', async () => {
  const requests: any[] = [];
  const { tools } = load();
  const ctx = {
    model: model('anthropic', 'claude-opus-5'),
    thinkingLevel: 'off',
    modelRegistry: modelRegistry([model('openai-codex', 'gpt-5.6-sol')], requests),
  };

  await tools
    .get('oracle')
    .execute('call', { prompt: 'Is this right?' }, undefined, undefined, ctx);

  assert.equal(requests[0]?.baseUrl, 'https://enterprise.example');
});

test('/oracle sends requests to the base URL resolved by authentication', async () => {
  const requests: any[] = [];
  const { commands } = load();
  const oracleModel = model('github-copilot', 'gpt-5');
  const theme = { fg: (_colour: string, text: string) => text, bold: (text: string) => text };
  const ctx = {
    mode: 'tui',
    thinkingLevel: 'off',
    waitForIdle: async () => {},
    sessionManager: {
      buildContextEntries: () => [
        {
          type: 'message',
          message: { role: 'user', content: 'Question?', timestamp: 1 },
        },
        {
          type: 'message',
          message: {
            role: 'assistant',
            provider: 'anthropic',
            model: 'claude-opus-5',
            stopReason: 'stop',
            content: [{ type: 'text', text: 'Answer.' }],
            timestamp: 2,
          },
        },
      ],
    },
    modelRegistry: modelRegistry([oracleModel], requests),
    ui: {
      notify: () => {},
      custom: async (factory: any) => {
        let result: unknown;
        const component = await factory(
          { requestRender: () => {} },
          theme,
          {},
          (value: unknown) => {
            result = value;
          },
        );
        if (component instanceof Object && 'onAbort' in component) {
          await new Promise((resolve) => setTimeout(resolve, 10));
          component.dispose?.();
          return result;
        }
        return { model: oracleModel, thinkingLevel: 'off' };
      },
    },
  };

  await commands.get('oracle').handler('', ctx);

  assert.equal(requests[0]?.baseUrl, 'https://enterprise.example');
});

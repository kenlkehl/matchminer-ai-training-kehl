const assert = require('node:assert/strict');
const http = require('node:http');
const { once } = require('node:events');
const { test } = require('node:test');
const { LlamaRuntime } = require('../electron/runtime/llama-runtime.cjs');

test('native request header follows the selected GGUF model, preserving explicit empty instructions', async () => {
  const requests = [];
  const server = http.createServer(async (request, response) => {
    let body = '';
    for await (const chunk of request) body += chunk;
    requests.push(JSON.parse(body));
    response.setHeader('Content-Type', 'application/json');
    response.end(JSON.stringify({ choices: [{ message: { content: 'Synthetic result.' } }] }));
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const runtime = new LlamaRuntime({});
  runtime.ensureStarted = async () => {};
  runtime.baseUrl = (route) => `http://127.0.0.1:${server.address().port}${route}`;
  try {
    for (const [repo, header] of [
      ['vendor/gpt-oss-120b-GGUF', 'Reasoning: high'],
      ['nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4', ''],
      ['google/gemma-4-31B-it', ''],
      ['ksg-dfci/OncoReasoning-0526-GGUF', ''],
    ]) {
      await runtime.generate({ prompt: 'Synthetic request.', llamaModelRepo: repo });
      assert.equal(requests.at(-1).messages[0].content, header);
      assert.equal(requests.at(-1).messages[1].content, 'Synthetic request.');
    }
    await runtime.generate({ prompt: 'Synthetic request.', repo: 'openai/gpt-oss-20b', systemPrompt: '' });
    assert.equal(requests.at(-1).messages[0].content, '');
  } finally {
    server.closeAllConnections();
    server.close();
    await once(server, 'close');
  }
});

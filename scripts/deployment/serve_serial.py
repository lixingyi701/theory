#!/usr/bin/env python3
"""Loopback adapter around synchronous offline vLLM, one prompt per generate.

A research-serving profile; no streaming, concurrency, retries or response cache.
Identical-payload reproducibility must be measured on the actual GPU/version.
"""
import argparse
import importlib.metadata
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PROFILE = 'offline_serial_eager_nocache_v1'


def validate_payload(payload, served):
    allowed = {'model', 'messages', 'max_tokens', 'temperature', 'top_p', 'top_k', 'min_p',
               'seed', 'stop', 'chat_template_kwargs'}
    if set(payload) - allowed or payload.get('model') != served:
        raise ValueError('Unsupported model or request fields')
    if type(payload.get('max_tokens')) is not int or not 1 <= payload['max_tokens'] <= 256:
        raise ValueError('Output budget must be an integer in 1..256')
    if not isinstance(payload.get('messages'), list) or not payload['messages']:
        raise ValueError('Text messages required')
    for m in payload['messages']:
        if set(m) != {'role', 'content'} or m['role'] not in {'system', 'user', 'assistant'} or not isinstance(m['content'], str):
            raise ValueError('Unsupported message')
    if payload.get('chat_template_kwargs') != {'enable_thinking': False}:
        raise ValueError('This profile is fixed to non-thinking')
    if type(payload.get('seed')) is not int:
        raise ValueError('Explicit per-request seed required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--port', type=int, default=8000)
    a = p.parse_args()
    if importlib.metadata.version('vllm') != '0.10.2':
        raise ValueError('Pinned vLLM0.10.2 required')
    receipt = json.loads((a.model_dir / 'download_receipt.json').read_text())
    if receipt['revision'] != 'b968826d9c46dd6066d109eabc6255188de91218':
        raise ValueError('Pinned Qwen3-8B revision required')
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['VLLM_NO_USAGE_STATS'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(a.model_dir), dtype='bfloat16', tensor_parallel_size=1,
              max_model_len=32768, max_num_seqs=1, max_num_batched_tokens=32768,
              enable_chunked_prefill=False, enable_prefix_caching=False,
              enforce_eager=True, seed=0, gpu_memory_utilization=0.85,
              disable_cascade_attn=True, generation_config='vllm')
    tokenizer = llm.get_tokenizer()
    served = 'qwen3-8b'

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, value):
            data = json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path == '/v1/models':
                self.send_json(200, {'object': 'list', 'serving_profile': PROFILE,
                                     'data': [{'id': served, 'object': 'model', 'max_model_len': 32768}]})
            else:
                self.send_json(404, {'error': 'Unknown endpoint'})
        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.send_json(404, {'error': 'Unknown endpoint'}); return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError('Invalid body size')
                body = json.loads(self.rfile.read(length))
                validate_payload(body, served)
                ids = tokenizer.apply_chat_template(body['messages'], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
                if len(ids) + body['max_tokens'] > 32768:
                    raise ValueError('Context budget exceeded; no truncation')
                params = {k: body[k] for k in ['temperature', 'top_p', 'top_k', 'min_p', 'seed', 'max_tokens', 'stop'] if k in body}
                output = llm.generate([{'prompt_token_ids': ids}], SamplingParams(**params), use_tqdm=False)[0]
                choice = output.outputs[0]
                self.send_json(200, {'id': f'local-{output.request_id}', 'object': 'chat.completion',
                    'created': int(time.time()), 'model': served, 'serving_profile': PROFILE,
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': choice.text},
                                 'finish_reason': choice.finish_reason}],
                    'usage': {'prompt_tokens': len(output.prompt_token_ids), 'completion_tokens': len(choice.token_ids),
                              'total_tokens': len(output.prompt_token_ids) + len(choice.token_ids)}})
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json(400, {'error': type(exc).__name__})
        def log_message(self, fmt, *args):
            # Requests/responses live in the runner journal, never in server logs.
            pass
    print(json.dumps({'ready': True, 'profile': PROFILE, 'context': 32768}), flush=True)
    HTTPServer(('127.0.0.1', a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()

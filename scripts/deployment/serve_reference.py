#!/usr/bin/env python3
"""Serial raw-completion endpoint for the unmodified author ReAct LLM contract."""
import argparse
import importlib.metadata
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PROFILE='upstream_react_raw_serial_v1'
REVISION='b968826d9c46dd6066d109eabc6255188de91218'


def validate(payload):
    fields={'model','prompt','temperature','top_p','max_tokens','frequency_penalty','presence_penalty','stop','seed'}
    if set(payload)!=fields or payload['model']!='qwen3-8b' or not isinstance(payload['prompt'],str):
        raise ValueError('Author completion contract required')
    for k,v in {'temperature':0,'top_p':1,'max_tokens':100,'frequency_penalty':0.0,'presence_penalty':0.0,'stop':['\n']}.items():
        if payload[k]!=v:
            raise ValueError('Author decoding settings changed')
    if type(payload['seed']) is not int:
        raise ValueError('Explicit per-request seed required')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--port',type=int,default=8000);a=p.parse_args()
    if importlib.metadata.version('vllm')!='0.10.2':
        raise ValueError('Pinned vLLM0.10.2 required')
    if json.loads((a.model_dir/'download_receipt.json').read_text())['revision']!=REVISION:
        raise ValueError('Pinned model required')
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1')
    from vllm import LLM,SamplingParams
    llm=LLM(model=str(a.model_dir),dtype='bfloat16',tensor_parallel_size=1,max_model_len=32768,
            max_num_seqs=1,max_num_batched_tokens=32768,enable_prefix_caching=False,
            enforce_eager=True,seed=0,gpu_memory_utilization=0.85,disable_cascade_attn=True,generation_config='vllm')
    tokenizer=llm.get_tokenizer()
    health={'object':'list','serving_profile':PROFILE,'model_revision':REVISION,
            'capabilities':['raw_completions'],'prompt_format':'raw text, no chat template or action repair',
            'data':[{'id':'qwen3-8b','object':'model','max_model_len':32768}]}
    class Handler(BaseHTTPRequestHandler):
        def send_json(self,status,value):
            data=json.dumps(value).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
        def do_GET(self):
            self.send_json(200,health) if self.path=='/v1/models' else self.send_json(404,{'error':'Unknown endpoint'})
        def do_POST(self):
            if self.path!='/v1/completions':
                self.send_json(404,{'error':'Raw completions only'});return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=1024*1024:
                    raise ValueError('Invalid body size')
                body=json.loads(self.rfile.read(size));validate(body)
                tokens=tokenizer.encode(body['prompt'],add_special_tokens=False)
                if len(tokens)+body['max_tokens']>32768:
                    raise ValueError('Context exhausted; no prompt truncation')
                params={k:body[k] for k in ['temperature','top_p','max_tokens','frequency_penalty','presence_penalty','stop','seed']}
                result=llm.generate([{'prompt_token_ids':tokens}],SamplingParams(**params),use_tqdm=False)[0]
                choice=result.outputs[0]
                self.send_json(200,{'id':f'local-{result.request_id}','object':'text_completion','created':int(time.time()),
                    'model':'qwen3-8b','serving_profile':PROFILE,
                    'choices':[{'index':0,'text':choice.text,'finish_reason':choice.finish_reason}],
                    'usage':{'prompt_tokens':len(tokens),'completion_tokens':len(choice.token_ids),
                             'total_tokens':len(tokens)+len(choice.token_ids)}})
            except (ValueError,TypeError,KeyError) as exc:
                self.send_json(400,{'error':type(exc).__name__})
        def log_message(self,*args):
            pass
    print(json.dumps({'ready':True,**health}),flush=True)
    HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Serial raw-completion service with two explicit, audited sampling profiles."""
import argparse
import importlib.metadata
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PROFILE = 'reflexion_raw_serial_profiles_v2'
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'

def validate(payload):
    fields = {'model','prompt','temperature','top_p','max_tokens','frequency_penalty','presence_penalty','stop','seed'}
    chosen = payload.get('sampling_profile','author_reflexion')
    if chosen=='qwen3_non_thinking_recommended_v1':
        fields |= {'sampling_profile','top_k','min_p','stop_token_ids'}
        if payload.get('temperature')!=.7 or payload.get('top_p')!=.8 or payload.get('top_k')!=20 or payload.get('min_p')!=0 or payload.get('stop_token_ids')!=[151645,151643]:
            raise ValueError('Official non-thinking sampling mismatch')
    elif chosen!='author_reflexion': raise ValueError('Unknown sampling profile')
    if set(payload) != fields or payload['model'] != 'qwen3-8b' or not isinstance(payload['prompt'], str):
        raise ValueError('Completion contract mismatch')
    if payload['max_tokens'] != 256 or payload['frequency_penalty'] != 0 or payload['presence_penalty'] != 0:
        raise ValueError('Decoding mismatch')
    if chosen=='author_reflexion' and (payload['top_p']!=1 or not any(abs(payload['temperature'] - i * .2) < 1e-8 for i in range(6))):
        raise ValueError('Temperature mismatch')
    if payload['stop'] not in (None, ['\n']) or type(payload['seed']) is not int:
        raise ValueError('Stop/seed mismatch')

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--port',type=int,default=8001);p.add_argument('--gpu-memory-utilization',type=float,default=.55)
    a=p.parse_args()
    if importlib.metadata.version('vllm') != '0.10.2': raise ValueError('vLLM revision mismatch')
    if json.loads((a.model_dir/'download_receipt.json').read_text())['revision'] != REVISION: raise ValueError('Model revision mismatch')
    generation=json.loads((a.model_dir/'generation_config.json').read_text())
    if generation['eos_token_id']!=[151645,151643]: raise ValueError('Model EOS configuration mismatch')
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1')
    from vllm import LLM, SamplingParams
    llm=LLM(model=str(a.model_dir),dtype='bfloat16',tensor_parallel_size=1,max_model_len=32768,
            max_num_seqs=1,max_num_batched_tokens=32768,enable_prefix_caching=False,enforce_eager=True,
            seed=0,gpu_memory_utilization=a.gpu_memory_utilization,disable_cascade_attn=True,generation_config='vllm')
    tokenizer=llm.get_tokenizer()
    health={'serving_profile':PROFILE,'model_revision':REVISION,'capabilities':['raw_completions'],
            'gpu_memory_utilization':a.gpu_memory_utilization,'data':[{'id':'qwen3-8b','max_model_len':32768}]}
    health['sampling_profiles']=['author_reflexion','qwen3_non_thinking_recommended_v1']
    health['service_sha256']=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()
    health['dtype']='bfloat16';health['tokenizer_eos_token_id']=tokenizer.eos_token_id
    health['model_protocol_sha256']=__import__('hashlib').sha256(Path('configs/reflexion_qwen3_8b_protocol.json').read_bytes()).hexdigest()
    class Handler(BaseHTTPRequestHandler):
        def send_json(self,status,value):
            data=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self): self.send_json(200,health) if self.path=='/v1/models' else self.send_json(404,{'error':'endpoint'})
        def do_POST(self):
            if self.path!='/v1/completions': self.send_json(404,{'error':'endpoint'});return
            try:
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<=1024*1024: raise ValueError('Body size')
                body=json.loads(self.rfile.read(size));validate(body)
                tokens=tokenizer.encode(body['prompt'],add_special_tokens=False)
                if len(tokens)+256>32768: raise ValueError('Context exhausted')
                kwargs={k:body[k] for k in ['temperature','top_p','max_tokens','frequency_penalty','presence_penalty','stop','seed']}
                kwargs.update(top_k=body.get('top_k',-1),min_p=body.get('min_p',0),ignore_eos=False)
                if 'stop_token_ids' in body: kwargs['stop_token_ids']=body['stop_token_ids']
                params=SamplingParams(**kwargs)
                effective={k:getattr(params,k) for k in kwargs}
                result=llm.generate([{'prompt_token_ids':tokens}],params,use_tqdm=False)[0];choice=result.outputs[0]
                self.send_json(200,{'id':f'local-{result.request_id}','model':'qwen3-8b','created':int(time.time()),
                    'serving_profile':PROFILE,'choices':[{'text':choice.text,'finish_reason':choice.finish_reason,'index':0}],
                    'sampling_profile':body.get('sampling_profile','author_reflexion'),
                    'effective_sampling_params':effective,
                    'usage':{'prompt_tokens':len(tokens),'completion_tokens':len(choice.token_ids),'total_tokens':len(tokens)+len(choice.token_ids)}})
            except (ValueError,TypeError,KeyError) as exc: self.send_json(400,{'error':str(exc)})
        def log_message(self,*args): pass
    print(json.dumps({'ready':True,**health}),flush=True)
    HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

if __name__=='__main__': main()

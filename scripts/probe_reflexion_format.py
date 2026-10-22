#!/usr/bin/env python3
"""Compare saved reflector queries with Qwen's official non-thinking template."""
import json
from pathlib import Path
from agent_reliability.io import read_json,write_json,digest,file_hash
from agent_reliability.llm import CallJournal
from agent_reliability.reference import CompletionTransport

def main():
    from transformers import AutoTokenizer
    model=Path('models/Qwen3-8B')
    if read_json(model/'download_receipt.json')['revision']!='b968826d9c46dd6066d109eabc6255188de91218': raise ValueError('Model changed')
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    output=Path('runs/reflexion-format-probe-20261022')
    manifest=read_json(Path('repro/baseline-20261019/dev12.json'));root=Path('runs/reflexion-dev12-20261022')
    sources=[]
    for task in manifest['tasks']:
        directory=root/'tasks'/task['task_id']/'trial-00'/'calls'
        found=sorted(directory.glob('reflector-*/response.json'))
        if found: sources.append((task['task_id'],found[0]))
        if len(sources)==6: break
    if len(sources)!=6: raise ValueError('Need six first-failure queries from distinct tasks')
    identity={'driver_sha256':file_hash(Path(__file__)),'template_sha256':digest(tokenizer.chat_template),
              'enable_thinking':False,'tasks':[t for t,_ in sources],
              'selection':'first six tasks in frozen dev12 order with a first-trial reflection',
              'scope':'reflector interface diagnosis; no agent task-success comparison'}
    if (output/'run.json').exists():
        if read_json(output/'run.json')!=identity: raise ValueError('Probe changed')
    else: write_json(output/'run.json',identity)
    transport=CompletionTransport('http://127.0.0.1:8001/v1','local-development',180)
    journal=CallJournal(output/'calls',transport,6);rows=[]
    for task_id,path in sources:
        source=read_json(path);payload=dict(read_json(path.parent/'attempt-000.request.json')['payload'])
        raw_prompt=payload['prompt']
        payload['prompt']=tokenizer.apply_chat_template([{'role':'user','content':raw_prompt}],tokenize=False,
                                                       add_generation_prompt=True,enable_thinking=False)
        result=journal.complete(task_id,payload)
        a=source['response'];b=result['response']
        rows.append({'task_id':task_id,'raw_payload_sha256':source['payload_sha256'],
                     'raw_finish_reason':a['choices'][0]['finish_reason'],'chat_finish_reason':b['choices'][0]['finish_reason'],
                     'raw_output_tokens':a['usage']['completion_tokens'],'chat_output_tokens':b['usage']['completion_tokens'],
                     'chat_nonempty':bool(b['choices'][0]['text'].strip())})
        write_json(output/'summary.json',{'scope':identity['scope'],'expected_queries':6,'completed_queries':len(rows),
                                         'rows':rows,'costs':journal.totals()})
        print(rows[-1],flush=True)

if __name__=='__main__': main()

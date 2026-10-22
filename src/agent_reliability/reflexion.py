"""Pinned author Reflexion functions with recorded model/environment I/O.

Actor, history and memory-update functions are compiled from verified sources.
The outer scheduler isolates tasks and checks native won against author done.
"""
import ast
import json
import sys
from pathlib import Path
from .io import digest, file_hash, read_json, seed_for, write_json
from .llm import CallJournal, InfrastructureError
from .reference import RecordedBatchEnvironment, official_environment

COMMIT='218cf0ef1df84b05ce379dd4a8e47f17766733a0'
ROOT=Path('third_party/reflexion')

def author_namespace(root=ROOT):
    receipt=read_json(root/'receipt.json')
    if receipt['commit']!=COMMIT: raise ValueError('Reflexion commit mismatch')
    for name,sha in receipt['files'].items():
        if file_hash(root/name)!=sha: raise ValueError(f'Reflexion source changed: {name}')
    ns={'sys':sys,'List':list,'Dict':dict,'Any':object,'Tuple':tuple,'Model':str,
        'FEW_SHOT_EXAMPLES':(root/'reflexion_few_shot_examples.txt').read_text()}
    for name in ['env_history.py','alfworld_trial.py','generate_reflections.py']:
        tree=ast.parse((root/name).read_text())
        selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        if name=='alfworld_trial.py':
            selected=[n for n in selected if n.name in {'llm','process_ob','alfworld_run'}]
            selected.extend(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PREFIXES' for t in n.targets))
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(root/name),'exec'),ns)
    return ns,receipt

class Events:
    def __init__(self,path): self.path=path
    def add(self,kind,**values):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a') as f: f.write(json.dumps({'event':kind,**values})+'\n')

class LoggedEnvironment(RecordedBatchEnvironment):
    def __init__(self,env,directory,checkpoint,events):
        super().__init__(env,directory,checkpoint);self.events=events
    def step(self,actions):
        value=super().step(actions)
        self.events.add('actor-action',index=len(self.transitions)-1,**self.last)
        return value

class CompletionIO:
    def __init__(self,journal,task,trial,seed,events,allowance,input_format='raw'):
        self.journal,self.task,self.trial,self.seed,self.events,self.allowance=journal,task,trial,seed,events,allowance
        self.index=0;self.role='actor'
        self.input_format=input_format
    def __call__(self,prompt,temperature=0.,max_tokens=256,stop_strs=None):
        logical=f'{self.role}-{self.index:04d}'
        if self.input_format in {'qwen3_non_thinking','qwen3_non_thinking_actionline'}:
            wrapper=read_json(Path('configs/qwen3_8b_non_thinking_wrapper.json'))
            prompt=wrapper['prefix']+prompt+wrapper['suffix']
        elif self.input_format!='raw': raise ValueError('Unknown input format')
        payload={'model':'qwen3-8b','prompt':prompt,'temperature':temperature,'top_p':1,'max_tokens':max_tokens,
                 'frequency_penalty':0.,'presence_penalty':0.,'stop':stop_strs,
                 'seed':seed_for(self.task['task_id'],self.seed,self.trial,self.index,self.role)}
        if not (self.journal.directory/logical/'response.json').exists(): self.allowance()
        saved=self.journal.complete(logical,payload);choice=saved['response']['choices'][0]
        text=choice['text']
        if not isinstance(text,str): raise InfrastructureError('Completion text missing')
        decoded=text
        if self.input_format=='qwen3_non_thinking_actionline' and self.role=='actor' and text.lstrip().startswith('> '):
            decoded=text.lstrip()[2:]
        write_json(self.journal.directory/logical/'validation.json',{'valid':bool(text.strip()),
            'truncated_output':choice.get('finish_reason')=='length','finish_reason':choice.get('finish_reason'),
            'decoder':'one_leading_action_marker' if decoded!=text else 'identity',
            'decoded_sha256':digest(decoded)})
        self.events.add('model-call',role=self.role,index=self.index,payload_sha256=digest(payload),temperature=temperature)
        self.index+=1
        return decoded

def run_attempt(task,memory,trial,directory,transport,data_root,seed,allowance,reflect=True,input_format='raw'):
    ns,receipt=author_namespace()
    identity={'task':task,'memory':memory,'trial':trial,'seed':seed,'upstream':receipt,
              'profile':'upstream_reflexion_raw_serial_v1','input_format':input_format}
    if input_format in {'qwen3_non_thinking','qwen3_non_thinking_actionline'}:
        identity['wrapper_sha256']=file_hash(Path('configs/qwen3_8b_non_thinking_wrapper.json'))
    identity['sha256']=digest(identity)
    if (directory/'identity.json').exists():
        if read_json(directory/'identity.json')!=identity: raise ValueError('Attempt identity changed')
    else: write_json(directory/'identity.json',identity)
    if (directory/'result.json').exists(): return read_json(directory/'result.json')
    checkpoint=read_json(directory/'checkpoint.json') if (directory/'checkpoint.json').exists() else None
    # Events are append-only execution records; model/env artifacts are canonical
    # on resume. Each replay segment is explicitly identified.
    events=Events(directory/'events.jsonl');events.add('execution-segment',replay=checkpoint is not None)
    journal=CallJournal(directory/'calls',transport,295)
    io=CompletionIO(journal,task,trial,seed,events,allowance,input_format)
    ns['get_completion']=io
    def no_chat(**kwargs): raise ValueError('This protocol selects the author completion pathway')
    ns['get_chat']=no_chat
    env=official_environment(data_root,task,seed,Path('third_party/react'))
    try:
        observations,_=env.reset();ob='\n'.join(observations[0].split('\n\n')[1:])
        demos=read_json(ROOT/'prompts/alfworld_3prompts.json')
        key=next(v for k,v in ns['PREFIXES'].items() if task['task_type'].startswith(k))
        prompt='Interact with a household to solve a task. Here are two examples.\n'+demos[f'react_{key}_1']+demos[f'react_{key}_0']
        batch=LoggedEnvironment(env,directory,checkpoint,events)
        events.add('restore',initial_observation_sha256=digest(ob),environment_seed=seed)
        events.add('memory-visible',memory_sha256=digest(memory[-3:]),entries=len(memory[-3:]))
        history,author_success=ns['alfworld_run'](batch,prompt,memory,to_print=False,ob=ob,model='text-davinci-003')
        if len(batch.transitions)<len(batch.committed): raise InfrastructureError('Replay ended before checkpoint')
        native_success=bool(batch.last and batch.last['won'])
        if bool(author_success)!=native_success:
            raise InfrastructureError('Author done success differs from native won; inspect trajectory')
        history_text=str(history)
        log='\n#####\n\nEnvironment #0:\n'+history_text+'\n\nSTATUS: '+('OK' if native_success else 'FAIL')+'\n\n#####\n'
        # Match run_trial's one-environment log, including the footer consumed
        # by the unchanged update_memory parser.
        log+=f"\n-----\nSUCCESS: {int(native_success)}\nADDITIONAL SUCCESS: {int(native_success)}\nFAIL: {int(not native_success)}\nTOTAL: 1\nACCURACY: {round(int(native_success),2)}\n-----"
        (directory/'trial.log').write_text(log)
        events.add('trial-end',success=native_success,decisions=len(batch.transitions),author_success=bool(author_success))
        config=[{'name':'env_0','memory':list(memory),'is_success':native_success,'skip':False}]
        if reflect:
            io.role='reflector';ns['update_memory'](str(directory/'trial.log'),config)
        after=config[0]['memory']
        if len(after)>len(memory): events.add('reflection-write',memory_sha256=digest(after),entries=len(after))
        status='success' if native_success else ('environment_failure' if batch.last and batch.last['done'] else 'repeat_or_decision_budget')
        result={'task_id':task['task_id'],'scene_id':task['scene_id'],'task_type':task['task_type'],'trial':trial,
            'success':native_success,'status':status,'decisions':len(batch.transitions),
            'memory_before':memory,'memory_after':after,'initial_observation_sha256':digest(ob),
            'trajectory_sha256':digest(batch.transitions),'identity_sha256':identity['sha256'],**journal.totals()}
        write_json(directory/'result.json',result)
        return result
    finally: env.close()

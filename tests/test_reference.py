"""Concrete regression against the author's control flow, not a benchmark."""
import copy
import tempfile
import unittest
from pathlib import Path
from agent_reliability.io import read_json
from agent_reliability.llm import InfrastructureError,UncertainRequest
from agent_reliability.reference import author_code,run_reference_episode


class Batch:
    def __init__(self):
        self.actions=[]
    def reset(self):
        self.actions=[]
        return ['-= Welcome =-\n\npublic observation\n\npublic goal'],{'won':[False],'admissible_commands':['ORACLE_ONLY']}
    def step(self,actions):
        self.actions+=actions;won=len(self.actions)==3
        return [f'feedback {len(self.actions)}'],[float(won)],[won],{'won':[won],'admissible_commands':['ORACLE_ONLY']}


class Provider:
    def __init__(self,fail=None):
        self.calls=[];self.fail=fail
    def complete(self,payload):
        self.calls.append(copy.deepcopy(payload))
        if len(self.calls)==self.fail:
            raise InfrastructureError('injected transport uncertainty')
        text=' think: I should act.' if 'think: I should act.' not in payload['prompt'] else ' go to desk 1'
        return {'choices':[{'text':text,'finish_reason':'stop'}],'usage':{'prompt_tokens':3,'completion_tokens':2}}


class ReferenceTests(unittest.TestCase):
    def test_original_thought_is_sent_to_world_and_observation_becomes_ok(self):
        ns,_=author_code(Path('third_party/react'));env=Batch();env.reset();prompts=[]
        def llm(prompt,stop):
            prompts.append(prompt);return ['think: plan','go to desk 1','take apple 1 from desk 1'][len(prompts)-1]
        ns['env']=env;ns['llm']=llm
        self.assertTrue(ns['alfworld_run']('examples\n',to_print=False,ob='goal'))
        self.assertEqual(env.actions[0],'think: plan')
        self.assertIn('think: plan\nOK.\n>',prompts[1])
        self.assertIn('go to desk 1\nfeedback 2\n>',prompts[2])
    def test_author_loop_has_49_calls_and_no_custom_truncation_failure(self):
        ns,_=author_code(Path('third_party/react'))
        class NeverDone:
            def step(self,actions):
                return ['Nothing happens.'],[0],[False],{'won':[False]}
        calls=[];ns['env']=NeverDone();ns['llm']=lambda prompt,stop:(calls.append(prompt) or 'think: continue')
        self.assertEqual(ns['alfworld_run']('',False,'goal'),0);self.assertEqual(len(calls),49)
    def test_cached_prefix_resume_and_no_oracle_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            config=read_json(Path('configs/reference_react_8b.json'));task={'task_id':'fake','scene_id':'fake','task_type':'pick_and_place_simple'}
            model={'served_name':'qwen3-8b'};d=Path(temp);p=Provider(fail=2)
            with self.assertRaises(InfrastructureError):
                run_reference_episode(Batch(),p,task,config,model,d)
            self.assertEqual(len(read_json(d/'checkpoint.json')['transitions']),1)
            with self.assertRaises(UncertainRequest):
                run_reference_episode(Batch(),p,task,config,model,d,resume=True)
            result=run_reference_episode(Batch(),p,task,config,model,d,resume=True,retry_uncertain=True)
            self.assertTrue(result['success']);self.assertEqual(result['dispatched_calls'],4)
            for call in p.calls:
                self.assertNotIn('ORACLE_ONLY',call['prompt']);self.assertNotIn('messages',call)
                self.assertEqual(call['max_tokens'],100)
            n=len(p.calls);run_reference_episode(Batch(),p,task,config,model,d,resume=True);self.assertEqual(len(p.calls),n)

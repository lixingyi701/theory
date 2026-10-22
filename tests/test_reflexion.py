"""Regression for author events/retries and recorded attempt reconstruction."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from agent_reliability.reflexion import author_namespace,run_attempt

class Env:
    def reset(self): self.n=0;return ['welcome\n\nPublic task'],{'won':[False]}
    def step(self,actions):
        self.n+=1
        return ['public feedback'],[0],[False],{'won':[False]}
    def close(self): pass

class Provider:
    def __init__(self): self.calls=[]
    def complete(self,payload):
        self.calls.append(payload)
        text='Plan: try again' if 'New plan:' in payload['prompt'] else 'go to desk 1'
        return {'choices':[{'text':text,'finish_reason':'stop'}],'usage':{'prompt_tokens':1,'completion_tokens':1}}

class ReflexionTests(unittest.TestCase):
    def test_author_short_retries_and_last_three_memory(self):
        ns,_=author_namespace();temps=[]
        def complete(**kwargs):
            temps.append(kwargs['temperature']);return '' if len(temps)<3 else 'think: plan'
        ns['get_completion']=complete
        self.assertEqual(ns['llm']('prompt',model='text-davinci-003'),'think: plan')
        self.assertEqual(temps,[0.,.2,.4])
        texts=[]
        ns['llm']=lambda prompt,**kwargs:(texts.append(prompt) or 'go to desk 1')
        env=Env();env.reset()
        ns['alfworld_run'](env,'demos',['OLD_MARKER','one','two','three'],False,'goal')
        self.assertNotIn('OLD_MARKER',texts[0]);self.assertIn('three',texts[0]);self.assertEqual(len(texts),2)
    def test_failed_trial_reflects_and_resume_does_not_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);provider=Provider();task={'task_id':'test','scene_id':'test','task_type':'pick_and_place_simple'}
            with patch('agent_reliability.reflexion.official_environment',return_value=Env()):
                a=run_attempt(task,[],0,d,provider,Path('.'),42,lambda:None)
                (d/'result.json').unlink()
                b=run_attempt(task,[],0,d,provider,Path('.'),42,lambda:None)
            self.assertEqual(a,b);self.assertEqual(len(provider.calls),3)
            self.assertEqual(a['memory_after'],['Plan: try again'])
            self.assertFalse(a['success']);self.assertEqual(a['decisions'],2)

if __name__=='__main__': unittest.main()

"""Client and server agree on the explicitly selected Qwen generation profile."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/deployment'))
from serve_reflexion_profiles import validate
from agent_reliability.io import read_json
from agent_reliability.llm import CallJournal
from agent_reliability.reflexion import CompletionIO, Events


class SamplingInterfaceTests(unittest.TestCase):
    def test_recommended_profile_wire_contract_and_raw_response(self):
        class Provider:
            def complete(self,payload):
                validate(payload)
                self.payload=payload
                return {'choices':[{'text':'> go to desk 1','finish_reason':'stop'}],
                    'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);provider=Provider();journal=CallJournal(root/'calls',provider,2)
            io=CompletionIO(journal,{'task_id':'example'},0,42,Events(root/'events.jsonl'),lambda:None,
                'qwen3_non_thinking_actionline','qwen3_non_thinking_recommended_v1')
            self.assertEqual(io('public task',temperature=0,stop_strs=['\n']),'go to desk 1')
            self.assertEqual(provider.payload['temperature'],.7)
            self.assertEqual(provider.payload['top_p'],.8)
            self.assertEqual(provider.payload['top_k'],20)
            self.assertEqual(read_json(root/'calls/actor-0000/response.json')['response']['choices'][0]['text'],'> go to desk 1')
            bad={**provider.payload,'top_p':1}
            with self.assertRaisesRegex(ValueError,'sampling mismatch'):
                validate(bad)


if __name__=='__main__':
    unittest.main()

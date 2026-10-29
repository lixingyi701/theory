"""The new native-text path must use the saved reflection, not its class label."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_middle_context_transport as prepare
from agent_reliability.io import digest,file_hash,write_json


class MiddleContextTests(unittest.TestCase):
    def test_native_text_is_hash_bound_and_kept_in_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cells/example/trial-01/result.json'
            text='Plan: Keep the object in inventory for heating. Other concrete advice is retained.'
            write_json(path,{'success':False,'memory_before':['previous'],'memory_after':['previous',text]})
            source={'cell_id':'example','result_sha256':file_hash(path),'reflection_sha256':digest(text),'label':'C'}
            protocol={'entries':{'N':'Plan: Continue.','C':'Plan: Short correction.'}}
            cell={'labels':['N','C','N'],'native_source':source}
            with patch.object(prepare,'SOURCE_RUN',directory):
                self.assertEqual(prepare.memory_for(protocol,cell),['Plan: Continue.',text,'Plan: Continue.'])
                cell['native_source']={**source,'reflection_sha256':digest('another text')}
                with self.assertRaises(ValueError):
                    prepare.memory_for(protocol,cell)


if __name__=='__main__':
    unittest.main()

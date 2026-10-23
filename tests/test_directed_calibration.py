"""Parallel workers must share one finite allowance and preserve FIFO alignment."""
import multiprocessing
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_directed_calibration import CalibrationStop, reserve_call
from agent_reliability.io import read_json
from prepare_directed_calibration import STATES


def spend(root):
    for _ in range(15):
        try:
            reserve_call(Path(root), {'deadline_unix': time.time()+60, 'max_calls': 17})
        except CalibrationStop:
            return


class DirectedCalibrationTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux', 'Server fcntl accounting')
    def test_parallel_workers_cannot_overspend(self):
        with tempfile.TemporaryDirectory() as directory:
            children = [multiprocessing.Process(target=spend, args=(directory,)) for _ in range(4)]
            for child in children:
                child.start()
            for child in children:
                child.join(10)
                self.assertEqual(child.exitcode, 0)
            self.assertEqual(read_json(Path(directory)/'budget.json')['reserved'], 17)
            with self.assertRaises(CalibrationStop):
                reserve_call(Path(directory), {'deadline_unix': time.time()-1, 'max_calls': 100})

    def test_deletion_and_real_entry_replacement_have_different_eviction(self):
        replace = STATES['recovery']['replace']
        delete = STATES['recovery']['delete']
        self.assertNotIn('H', (replace+['new'])[-3:])
        self.assertIn('H', (delete+['new'])[-3:])
        self.assertEqual(sorted(STATES['age']['oldest']), sorted(STATES['age']['newest']))
        self.assertNotIn('E', (STATES['age']['oldest']+['new'])[-3:])
        self.assertIn('E', (STATES['age']['newest']+['new'])[-3:])


if __name__ == '__main__':
    unittest.main()

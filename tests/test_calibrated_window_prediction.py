"""Independent first-step and FIFO-alignment checks for the new forecast recursion."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from predict_calibrated_finite_window import bounds,remaining,restore


class CalibratedWindowTests(unittest.TestCase):
    def test_constant_success_and_semantic_ambiguity(self):
        k={'S':.3,'E':.1,'C':.15,'U':.15,'A':.3}
        for b in range(9):
            for r in range(4):
                lo,hi=bounds({False:k,True:k},b,r)
                self.assertAlmostEqual(lo,1-.7**b);self.assertAlmostEqual(hi,1-.7**b)

    def test_deletion_delays_the_other_old_error(self):
        clean={'S':.7,'E':0.,'C':0.,'U':.3,'A':0.}
        error={'S':.1,'E':0.,'C':0.,'U':.9,'A':0.}
        history=['E','N','E']; corrected=restore(history,'correct_latest'); deleted=restore(history,'delete_latest')
        self.assertEqual((remaining(corrected),remaining(deleted)),(1,2))
        self.assertAlmostEqual(bounds({False:clean,True:error},2,remaining(corrected))[0],.73)
        self.assertAlmostEqual(bounds({False:clean,True:error},2,remaining(deleted))[0],.19)

    def test_full_history_enumeration_matches_compressed_state(self):
        kernels={False:{'S':.6,'E':.1,'C':.1,'U':.2,'A':0.},True:{'S':.2,'E':.5,'C':.1,'U':.2,'A':0.}}
        def full(history,b):
            if b==0:return 0.
            k=kernels['E' in history]
            return k['S']+sum(k[x]*full((history+[x])[-3:],b-1) for x in ['E','C','U'])
        for history in [[],['N'],['E'],['E','N'],['N','E','N'],['E','N','E'],['C','N','N']]:
            for b in range(5):
                lo,hi=bounds(kernels,b,remaining(history))
                self.assertAlmostEqual(lo,full(history,b));self.assertAlmostEqual(hi,lo)


if __name__=='__main__':unittest.main()

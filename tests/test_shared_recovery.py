import unittest
import numpy as np
from shared_recovery_model import features,success_objective,fit,component_kernel,budget_probability,PRIOR


class SharedRecoveryTests(unittest.TestCase):
    def test_gradient_and_finite_fit(self):
        states=[[{'type':k,'native':False}] for k in ['E','C','N']]
        rows=[{'state':states[i%3],'task_id':str(i//3),'outcome':outcome}
              for i,outcome in enumerate(['E','S','U','E','C','S','B','S','A'])]
        x=np.asarray([features(r['state'],'information_order') for r in rows]);y=np.asarray([r['outcome']=='S' for r in rows],float)
        groups=[np.arange(i,i+3) for i in range(0,9,3)];b=np.linspace(-.2,.2,x.shape[1])
        _,gradient=success_objective(b,x,y,groups,1.5)
        numeric=[]
        for i in range(len(b)):
            delta=np.eye(len(b))[i]*1e-6
            numeric.append((success_objective(b+delta,x,y,groups,1.5)[0]-success_objective(b-delta,x,y,groups,1.5)[0])/2e-6)
        np.testing.assert_allclose(gradient,numeric,atol=1e-6)
        model=fit(rows,'information_order',1.5)
        np.testing.assert_allclose(component_kernel(model,states[0]).sum(axis=1),1.,atol=1e-12)

    def test_persistent_task_mixture_and_order(self):
        model={'family':'constant','scale':1.5,'success_coefficients':[0.],
               'write_coefficients_reference_A':[[0.,0.,0.,0.]]}
        p=component_kernel(model,[] )[:,0]
        self.assertAlmostEqual(budget_probability(model,[],2),float(PRIOR@(1-(1-p)**2)))
        self.assertGreater(abs(budget_probability(model,[],2)-(1-(1-PRIOR@p)**2)),.01)
        a=[{'type':k,'native':False} for k in ['E','N','C']]
        b=[{'type':k,'native':False} for k in ['C','N','E']]
        self.assertNotEqual(features(a,'information_order'),features(b,'information_order'))
        self.assertEqual(features([{'type':'B','native':True}],'information_order')[6],1.)


if __name__=='__main__': unittest.main()

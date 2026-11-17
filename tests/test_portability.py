import importlib,pkgutil,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from _common import load_inputs
from enose_drift.tcss import _tail_weights,group_proximal,GroupSparseCVaRMultinomial
from enose_drift.v211_classwise_gate import classwise_recall_weights,apply_classwise_gate
class PortableTests(unittest.TestCase):
    def test_import_all_numerical_modules(self):
        for p in (ROOT/'src/enose_drift').glob('*.py'):importlib.import_module('enose_drift.'+p.stem)
    def test_input_alignment(self):
        a,m=load_inputs();self.assertEqual(a['X_predict'].shape,(584,440));self.assertEqual(m.groupby('split').date.nunique().to_dict(),{'test':8,'train':24,'validation':7})
    def test_tail_weights(self):
        np.testing.assert_allclose(_tail_weights(np.arange(6),0),np.ones(6)/6)
        np.testing.assert_allclose(_tail_weights(np.arange(6),.75),[0,0,0,0,1/3,2/3])
    def test_group_prox_does_not_penalize_context(self):
        W=np.array([[3.],[4.],[9.]])
        np.testing.assert_allclose(group_proximal(W,[np.array([0,1])],5),[[0],[0],[9]])
    def test_gate_normalization(self):
        stack=np.array([[[.8,.2],[.3,.7]],[[.1,.9],[.6,.4]]])
        weights,_=classwise_recall_weights(stack,['a','b'],['a','b'],10)
        np.testing.assert_allclose(weights.sum(0),1)
        out=apply_classwise_gate(stack,weights);np.testing.assert_allclose(out.sum(1),1);self.assertTrue((out>=0).all())
    def test_selector_smoke_both_objectives(self):
        rng=np.random.default_rng(7);X=rng.normal(size=(24,6));y=np.tile(['a','b'],12);env=np.repeat(np.arange(6),4)
        for alpha in [0.,.75]:
            model=GroupSparseCVaRMultinomial({1:[0,1],2:[2,3]},alpha=alpha,max_iter=8).fit(X,y,env)
            self.assertTrue(np.isfinite(model.coef_).all());self.assertTrue(np.all(np.diff(model.objective_history_)<=1e-9))
    def test_single_class_metric_convention(self):
        from reproduce_results import metrics
        from _common import LABELS,PROBABILITY_COLUMNS
        d=pd.DataFrame({'y_true':[LABELS[0]],'y_pred':[LABELS[0]],**dict(zip(PROBABILITY_COLUMNS,[[1.],[0.],[0.],[0.],[0.]]))})
        result=metrics(d);self.assertEqual(result['mcc'],0.);self.assertAlmostEqual(result['macro_f1'],.2)
    def test_prediction_bundle_output_serialization(self):
        from enose_drift.evaluation import PredictionBundle
        from enose_drift.preparation import _bundle_frames
        bundle=PredictionBundle('test','sample','model','test',11,np.array(['s1','s2']),np.array(['d1','d1']),np.array(['a','b']),np.array(['a','b']),np.array([[.8,.2],[.1,.9]]),metadata={'availability':np.array([True,False])})
        predictions,dates=_bundle_frames([bundle]);self.assertEqual(len(predictions),2);self.assertEqual(len(dates),1);self.assertEqual(predictions.availability.iloc[0],'[true,false]')
if __name__=='__main__':unittest.main()

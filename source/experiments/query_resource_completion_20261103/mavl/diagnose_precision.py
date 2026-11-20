"""Compare old cuDNN default vs disabled TF32 on historical support only."""
import json
import os
from pathlib import Path
import numpy as np
import torch
import evaluate as e

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
    torch.set_num_threads(4);torch.manual_seed(1234);np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32=False
    root=e.HERE/'diagnostics'
    root.mkdir(exist_ok=True)
    test,support,sources=e.inputs('chest14')
    historical=e.ROOT/'baselines/runs/query_transfer_20261022_224826/MAVL_full_1/chest14'
    old=json.loads((historical/'query_adapter.json').read_text())
    oldpred=np.load(historical/'predictions.npz',allow_pickle=False)
    report={}; arrays={}
    with torch.inference_mode():
        native=e.Adapter('cuda')
        for flag in (False,True):
            torch.backends.cudnn.allow_tf32=flag
            query,fit,features=e.fit_and_capture(native,test,support,8)
            label='enabled' if flag else 'disabled'
            report[label]={'cudnn_allow_tf32':flag,'matmul_allow_tf32':False,'dtype':'float32','queries':{}}
            arrays[label+'_support_features']=features
            for key in ('original_queries','refined_queries'):
                actual=np.asarray(fit[key],np.float32);expected=np.asarray(old[key],np.float32)
                diff=np.abs(actual-expected)
                report[label]['queries'][key]=dict(max_abs_error=float(diff.max()),mean_abs_error=float(diff.mean()),
                    bitwise_equal=bool(np.array_equal(actual,expected)),
                    failed_elements=int(np.count_nonzero(diff>2e-6+1e-5*np.abs(expected))),
                    fixed_tolerance_passed=bool(np.allclose(actual,expected,atol=2e-6,rtol=1e-5)))
                arrays[label+'_'+key]=actual
            paths=[r['image'] for r in test['rows'][:8]]
            pred=e.run_views(query,paths,test['class_names'])
            report[label]['predictions']={}
            for branch in e.BRANCHES:
                expected=oldpred[branch][:8];actual=pred[branch]
                report[label]['predictions'][branch]=dict(max_abs_error=float(np.abs(actual-expected).max()),
                    bitwise_equal=bool(np.array_equal(actual,expected)),
                    fixed_tolerance_passed=bool(np.allclose(actual,expected,atol=2e-4,rtol=2e-4)))
                arrays[label+'_'+branch]=actual
            e.write(root/'precision_diagnosis.json',report)
            print(label,json.dumps(report[label]),flush=True)
    np.savez_compressed(root/'precision_diagnosis_arrays.npz',**arrays)
    assert report['enabled']['queries']['refined_queries']['fixed_tolerance_passed']
    assert all(x['fixed_tolerance_passed'] for x in report['enabled']['predictions'].values())

if __name__=='__main__':main()

"""Reproduce assignment Q2. Run with ../.venv-q2/bin/python run_experiment.py."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import platform
import time
import subprocess
import warnings
import numpy as np
import scipy
import sklearn
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]
COMMON = dict(tol=1e-3, shrinking=True, probability=False,
              class_weight=None, cache_size=512, max_iter=-1,
              decision_function_shape="ovr", break_ties=False, random_state=None)

def save(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2))
    temp.replace(path)

def candidate(config):
    X, y = load_svmlight_file(str(DATA / 'a9a'), n_features=123)
    # Dense conversion changes storage only, not the supplied feature values.
    X = X.toarray()
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    key = config['kernel'] + '_C' + str(config['C']) + '_g' + str(config.get('gamma', 'na'))
    result = dict(parameters=config, folds=[])
    start = time.monotonic()
    for fold, (train, valid) in enumerate(folds.split(X, y), 1):
        model = SVC(**COMMON, **config)
        t = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(X[train], y[train])
        pred = model.predict(X[valid])
        result['folds'].append(dict(fold=fold, train_size=len(train), validation_size=len(valid),
            correct=int(np.sum(pred == y[valid])), accuracy=float(accuracy_score(y[valid], pred)),
            fit_status=int(model.fit_status_), iterations=model.n_iter_.tolist(),
            warnings=[str(w.message) for w in caught], seconds=time.monotonic()-t))
        save(RESULTS / (key + '.json'), result)
        print(f"FOLD {key} {fold}/3 accuracy={result['folds'][-1]['accuracy']:.8f}", flush=True)
        if model.fit_status_ != 0 or caught:
            raise RuntimeError(f'Fit warning or non-convergence: {key}')
    scores = [f['accuracy'] for f in result['folds']]
    result.update(mean_accuracy=float(np.mean(scores)), std_accuracy=float(np.std(scores)),
                  seconds=time.monotonic()-start)
    save(RESULTS / (key + '.json'), result)
    return result

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in ['a9a', 'a9a.t']:
        path = DATA / name
        if not path.exists():
            temp = path.with_suffix('.download')
            subprocess.run(['curl', '--fail', '--location', '--retry', '2', '--max-time', '60',
                'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/' + name,
                '-o', str(temp)], check=True)
            temp.replace(path)
    X, y = load_svmlight_file(str(DATA/'a9a'), n_features=123)
    assert X.shape == (32561,123) and set(y) == {-1,1}
    assert set(X.data) == {1.0}
    splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X,y))
    np.savez_compressed(RESULTS/'cv_splits.npz', **{f'validation_{i+1}':v for i,(_,v) in enumerate(splits)})
    metadata = dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
        scipy=scipy.__version__, training_shape=list(X.shape), training_labels={str(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))},
        cv=dict(n_splits=3,shuffle=True,random_state=42), svc_common=COMMON,
        preprocessing='None; supplied binary features converted from CSR to dense storage.',
        sha256={n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['a9a','a9a.t']})
    save(RESULTS/'metadata.json', metadata)
    print('METADATA',json.dumps(metadata),flush=True)
    configs = [dict(kernel='linear',C=c) for c in VALUES]
    configs += [dict(kernel='rbf',C=c,gamma=g) for c in VALUES for g in VALUES]
    all_results=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(candidate,c) for c in configs]
        for job in as_completed(jobs):
            result=job.result()
            all_results.append(result)
            print('COMPLETE',len(all_results),'/30',result['parameters'],result['mean_accuracy'],flush=True)
    # Select using full precision CV scores only. Test labels have not been loaded.
    all_results.sort(key=lambda r: configs.index(r['parameters']))
    best=max(all_results,key=lambda r:r['mean_accuracy'])
    save(RESULTS/'cv_results.json',all_results)
    save(RESULTS/'selected_config.json',best)
    print('SELECTED',best['parameters'],flush=True)
    final=SVC(**COMMON,**best['parameters'])
    start=time.monotonic()
    final.fit(X.toarray(),y)
    assert final.fit_status_ == 0
    Xt,yt=load_svmlight_file(str(DATA/'a9a.t'), n_features=123)
    assert Xt.shape == (16281,123)
    pred=final.predict(Xt.toarray())
    np.save(RESULTS/'test_predictions.npy',pred)
    report=dict(parameters=best['parameters'],cv_accuracy=best['mean_accuracy'],
        test_shape=list(Xt.shape),correct=int(np.sum(pred==yt)),total=len(yt),
        test_accuracy=float(accuracy_score(yt,pred)),confusion_matrix=confusion_matrix(yt,pred,labels=[-1,1]).tolist(),
        confusion_matrix_label_order=[-1,1],fit_status=int(final.fit_status_),
        iterations=final.n_iter_.tolist(),seconds=time.monotonic()-start)
    save(RESULTS/'final_test.json',report)
    print('FINAL',json.dumps(report),flush=True)

if __name__ == '__main__':
    main()

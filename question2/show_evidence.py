"""Display and validate saved experimental records; does not retrain models."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
def read(name):
    return json.loads((RESULTS / name).read_text())

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('section', nargs='?', default='cv', choices=['cv', 'test'],
                    help='Which saved results to show (default: cv).')
args = parser.parse_args()
rows = read('cv_results.json')
meta = read('metadata.json')
final = read('final_test.json')
assert len(rows) == 30
for row in rows:
    assert len(row['folds']) == 3
    for f in row['folds']:
        assert f['fit_status'] == 0 and not f['warnings']
        assert abs(f['accuracy'] - f['correct']/f['validation_size']) < 1e-12
    assert abs(row['mean_accuracy'] - sum(f['accuracy'] for f in row['folds'])/3) < 1e-12
best = max(rows, key=lambda r:r['mean_accuracy'])
assert best['parameters'] == final['parameters']
print('Q2: VERIFIED SAVED EXPERIMENT RESULTS (no retraining)')
print('Source:', RESULTS)
if args.section == 'cv':
    print(f"Python {meta['python']} | scikit-learn {meta['sklearn']}")
    print('Training: 32561 x 123; no additional preprocessing')
    print('StratifiedKFold: 3 folds, shuffle=True, random_state=42')
    print('Checks passed: 30 configurations; 90 fits; no convergence warnings')
    print('\nLINEAR KERNEL: mean validation accuracy (%)')
    print('       C     0.01     0.05      0.1      0.5        1')
    linear = [r for r in rows if r['parameters']['kernel']=='linear']
    print('Accuracy '+''.join(f"{r['mean_accuracy']*100:9.4f}" for r in linear))
    print('\nRBF KERNEL: mean validation accuracy (%)')
    print(' C / gamma    0.01     0.05      0.1      0.5        1')
    for c in [.01,.05,.1,.5,1.]:
        selected = [r for r in rows if r['parameters']['kernel']=='rbf' and r['parameters']['C']==c]
        print(f'{c:10g}'+''.join(f"{r['mean_accuracy']*100:9.4f}" for r in selected))
    print('\nSelected by unrounded CV accuracy:', best['parameters'])
    print(f"Best mean CV accuracy: {best['mean_accuracy']*100:.4f}%")
else:
    import numpy as np
    from sklearn.datasets import load_svmlight_file
    _, labels = load_svmlight_file(str(ROOT/'data/a9a.t'), n_features=123)
    pred = np.load(RESULTS/'test_predictions.npy')
    assert pred.shape == labels.shape
    correct = int(np.sum(pred == labels))
    assert correct == final['correct'] and len(labels) == final['total']
    assert abs(correct/len(labels)-final['test_accuracy']) < 1e-12
    log = (ROOT/'experiment.log').read_text().splitlines()
    assert sum(line.startswith('COMPLETE ') for line in log)==30
    log_final=json.loads(next(line[len('FINAL '):] for line in log if line.startswith('FINAL ')))
    assert log_final==final
    print('Original execution log: experiment.log')
    print(next(line for line in log if line.startswith('COMPLETE 30 ')))
    print(next(line for line in log if line.startswith('SELECTED ')))
    print('\nFINAL EVALUATION (after fitting the complete training set)')
    print('Training examples: 32561 | Test examples: 16281 | Features: 123')
    print('Selected parameters:', final['parameters'])
    print(f"Mean CV accuracy: {final['cv_accuracy']*100:.4f}%")
    print(f"Correct test predictions: {correct} / {len(labels)}")
    print(f"TEST ACCURACY: {final['test_accuracy']*100:.4f}%")
    print('Final solver fit_status:', final['fit_status'])
    print('Confusion matrix (rows=true, columns=predicted; labels=[-1,+1]):')
    for row in final['confusion_matrix']:
        print(' ', row)
    print('\nPASS: saved predictions match test labels, reported accuracy, and original log.')

"""Verify saved replay predictions against source fingerprints and competition IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    parser.add_argument('--output-dir',type=Path,default=Path('runs/replay'))
    args=parser.parse_args()
    metrics=json.loads((args.output_dir/'metrics.json').read_text())
    test=pd.read_csv(args.data_dir/'test.csv',dtype={'PassengerId':str})
    train=pd.read_csv(args.data_dir/'train.csv',dtype={'PassengerId':str})
    oof=pd.read_csv(args.output_dir/'meta_oof_predictions.csv',dtype={'PassengerId':str})
    scores=pd.read_csv(args.output_dir/'model_comparison.csv').set_index('model')
    probability=pd.read_csv(args.output_dir/'test_probabilities.csv',dtype={'PassengerId':str})
    assert probability.PassengerId.equals(test.PassengerId)
    assert oof.PassengerId.equals(train.PassengerId)
    assert np.array_equal(oof.Actual.to_numpy(),train.Transported.astype(int).to_numpy())
    for name in scores.index:
        assert oof[name].between(0,1).all() and probability[name].between(0,1).all()
        observed=((oof[name]>=.5).astype(int)==oof.Actual).mean()
        assert np.isclose(observed,scores.loc[name,'development_meta_oof_accuracy'])
    for name,info in metrics['submissions'].items():
        path=args.output_dir/info['file']
        observed=hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed==info['expected_source_sha256'],f'{name} differs from the reference. Review versions and numerical changes.'
        submission=pd.read_csv(path,dtype={'PassengerId':str})
        assert submission.columns.tolist()==['PassengerId','Transported']
        assert submission.PassengerId.equals(test.PassengerId)
        assert len(submission)==4277 and not submission.isna().any().any()
        assert np.array_equal(submission.Transported.to_numpy(),probability[name]>=.5)
    assert ((probability.tabpfn_baseline>=.5)!=(probability.tabpfn_catboost_blend>=.5)).sum()==7
    print('Verified both source submission hashes, 4,277 IDs, boolean labels, OOF scores and 7 differing predictions.')

if __name__=='__main__':
    main()

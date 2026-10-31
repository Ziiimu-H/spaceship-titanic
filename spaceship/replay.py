"""Re-fit meta models over attributed public predictions; no TabPFN training."""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / '.cache/tabpfn'
OUT = ROOT / 'runs/replay'
MEMBERS = ['frozen', 'ft', 'ftrefit', 'ft_long_ne2']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logit(probability):
    p = np.clip(np.asarray(probability), 1e-5, 1-1e-5)
    return np.log(p/(1-p))


def make_meta(spline=False):
    lr = LogisticRegression(C=1, max_iter=3000)
    if spline:
        return make_pipeline(SplineTransformer(n_knots=3, degree=2, extrapolation='linear'),
                             StandardScaler(), lr)
    return lr


def main():
    global REF, OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--cache-dir',type=Path,default=REF)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    DATA=args.data_dir; REF=args.cache_dir; OUT=args.output_dir
    OUT.mkdir(parents=True,exist_ok=True)
    if not (REF/'provenance.json').exists():
        raise FileNotFoundError('Run python -m spaceship.assets first to download the reviewed public probabilities.')
    old_hashes={name:sha256(DATA/name) for name in ['train.csv','test.csv','sample_submission.csv']}
    provenance = json.loads((REF/'provenance.json').read_text())
    source_hashes = json.loads((REF/'asset_hashes.json').read_text())
    assert sha256(REF/'source_notebook.ipynb') == provenance['source_notebook_sha256']
    for name, checksum in source_hashes.items():
        assert sha256(REF/name) == checksum, name
    expected_data = {'train.csv':'17336d553f49ebdf6ecb266d2b5d3746e5dd308445f7c7864141c4f28d2a88d0',
                     'test.csv':'3c02b0703423d27d51a657a111b15a37b8d03b3c1de325e3a1e6b26f6572fc88'}
    for name, checksum in expected_data.items():
        assert sha256(DATA/name) == checksum, name
    train = pd.read_csv(DATA/'train.csv', dtype={'PassengerId':str})
    test = pd.read_csv(DATA/'test.csv', dtype={'PassengerId':str})
    sample = pd.read_csv(DATA/'sample_submission.csv', dtype={'PassengerId':str})
    oof = pd.read_csv(REF/'reference/kaggle_dataset/member_oof_seed42.csv', dtype={'PassengerId':str})
    target = pd.read_csv(REF/'reference/kaggle_dataset/member_test_seed42.csv', dtype={'PassengerId':str})
    assert train.PassengerId.equals(oof.PassengerId)
    assert test.PassengerId.equals(target.PassengerId) and test.PassengerId.equals(sample.PassengerId)
    assert train.PassengerId.is_unique and test.PassengerId.is_unique
    y = train.Transported.astype(int).to_numpy()
    group = train.PassengerId.str.split('_').str[0]
    fold = oof.fold.to_numpy()
    assert set(fold) == {1,2,3,4,5}
    assert pd.DataFrame({'group':group,'fold':fold}).groupby('group').fold.nunique().max() == 1
    shared_groups = len(set(group) & set(test.PassengerId.str.split('_').str[0]))
    assert shared_groups == 0
    assert hashlib.sha256(','.join(map(str,fold)).encode()).hexdigest() == '8b8d8882968f5a1cb2a8d0b68cfbff1f41d82be1ebaff441aff69fcfc41effca'
    for data in [oof[MEMBERS], target[MEMBERS]]:
        assert np.isfinite(data.to_numpy()).all()
        assert ((data >= 0) & (data <= 1)).all().all()
    with np.load(REF/'results/cat_context_d8.npz', allow_pickle=False) as cached_cat:
        cat_oof, cat_test = cached_cat['oof'].copy(), cached_cat['test'].copy()
    assert cat_oof.shape==(len(train),) and cat_test.shape==(len(test),)
    assert np.isfinite(cat_oof).all() and np.isfinite(cat_test).all()
    assert ((cat_oof>=0)&(cat_oof<=1)).all() and ((cat_test>=0)&(cat_test<=1)).all()

    z, zt = logit(oof[MEMBERS]), logit(target[MEMBERS])
    matrices = {
        'tabpfn_baseline': (z, zt, False),
        'tabpfn_spline': (z, zt, True),
        'tabpfn_cat_logistic': (np.c_[z,logit(cat_oof)], np.c_[zt,logit(cat_test)], False),
    }
    meta_oof, meta_test, fold_rows = {}, {}, []
    for name, (x,xt,spline) in matrices.items():
        probability = np.full(len(y),np.nan)
        for k in range(1,6):
            tr,va = np.flatnonzero(fold!=k), np.flatnonzero(fold==k)
            model = make_meta(spline)
            model.fit(x[tr],y[tr])
            probability[va] = model.predict_proba(x[va])[:,1]
            accuracy = float(((probability[va]>=.5)==y[va]).mean())
            fold_rows.append({'model':name,'fold':k,'accuracy':accuracy,'n_valid':len(va)})
            print(f'{name} fold {k}: {accuracy:.4%}',flush=True)
        assert np.isfinite(probability).all()
        final = make_meta(spline)
        final.fit(x,y)
        meta_oof[name] = probability
        meta_test[name] = final.predict_proba(xt)[:,1]
        joblib.dump(final, OUT/f'{name}_meta.joblib')
    meta_oof['tabpfn_catboost_blend'] = (meta_oof['tabpfn_spline']+meta_oof['tabpfn_cat_logistic'])/2
    meta_test['tabpfn_catboost_blend'] = (meta_test['tabpfn_spline']+meta_test['tabpfn_cat_logistic'])/2
    for k in range(1,6):
        va = fold==k
        fold_rows.append(dict(model='tabpfn_catboost_blend',fold=k,
            accuracy=float(((meta_oof['tabpfn_catboost_blend'][va]>=.5)==y[va]).mean()),n_valid=int(va.sum())))
    pd.DataFrame({'PassengerId':train.PassengerId,'Actual':y,'fold':fold,**meta_oof}).to_csv(OUT/'meta_oof_predictions.csv',index=False)
    pd.DataFrame({'PassengerId':test.PassengerId,**meta_test}).to_csv(OUT/'test_probabilities.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(OUT/'fold_scores.csv',index=False)
    summary = pd.DataFrame([dict(model=name,development_meta_oof_accuracy=float(((p>=.5)==y).mean())) for name,p in meta_oof.items()])
    summary.to_csv(OUT/'model_comparison.csv',index=False)
    submissions = {}
    for name in ['tabpfn_baseline','tabpfn_catboost_blend']:
        file = OUT/f'submission_{name}.csv'
        submission = pd.DataFrame({'PassengerId':test.PassengerId,'Transported':meta_test[name]>=.5})
        assert submission.columns.tolist()==sample.columns.tolist()
        assert len(submission)==4277 and not submission.isna().any().any()
        submission.to_csv(file,index=False,lineterminator='\n')
        pd.testing.assert_frame_equal(submission,pd.read_csv(file,dtype={'PassengerId':str}))
        observed = sha256(file)
        expected = provenance['expected_submission_hashes'][name]
        submissions[name] = {
            'file':file.name,'sha256':observed,'expected_source_sha256':expected,
            'matches_source_exactly':observed==expected,
            'source_reported_public_accuracy':provenance['reported_scores_not_measured_locally'][name],
            'different_predictions_vs_v2_in_original_project':{'tabpfn_baseline':296,'tabpfn_catboost_blend':301}[name],
        }
        print(name,submissions[name],flush=True)
    changed = (meta_test['tabpfn_baseline']>=.5)!=(meta_test['tabpfn_catboost_blend']>=.5)
    pd.DataFrame({'PassengerId':test.PassengerId[changed],
        'baseline_probability':meta_test['tabpfn_baseline'][changed],
        'blend_probability':meta_test['tabpfn_catboost_blend'][changed]}).to_csv(OUT/'baseline_vs_blend_changes.csv',index=False)
    report = {
        'train_rows':len(train),'test_rows':len(test),'threshold':.5,'train_test_shared_groups':shared_groups,
        'source_group_folds_verified':True,'public_asset_hashes_verified':len(source_hashes),
        'submissions':submissions,'baseline_blend_different_predictions':int(changed.sum()),
        'versions':dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__),
        'protected_file_hashes':old_hashes,'source_asset_hashes':source_hashes,
        'source_notebook_sha256':provenance['source_notebook_sha256'],
        'code_sha256':sha256(Path(__file__)),
        'limitations':[
            'Reused third-party public TabPFN and CatBoost probabilities; foundation models were not trained or independently audited locally.',
            'Source folds do not split passenger groups. Meta-model OOF over fixed base-model OOF is development validation, not fully nested evaluation.',
            'Source public scores are author-reported; our Kaggle scores require actual submissions.',
            'Do not compare these group-fold meta scores directly with earlier random-stratified OOF scores.',
        ],
    }
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    assert all(sha256(DATA/name)==checksum for name,checksum in old_hashes.items())
    fig,ax = plt.subplots(figsize=(8,4))
    names=['tabpfn_baseline','tabpfn_catboost_blend']
    vals=[float(summary.set_index('model').loc[n,'development_meta_oof_accuracy'])*100 for n in names]
    bars=ax.barh(['Four TabPFN / logistic','TabPFN + CatBoost / blend'],vals,color='#3574a5')
    ax.invert_yaxis(); ax.set_xlim(0,100)
    ax.set_xlabel('Development meta-OOF accuracy (%)')
    ax.set_title('Original source group folds / fixed public base predictions')
    ax.bar_label(bars,fmt='%.2f%%',padding=4)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT/'replay_comparison.png',dpi=160); plt.close(fig)
    print(summary.to_string(index=False),flush=True)
    print('Predictions changed between candidates:',int(changed.sum()),flush=True)
    print('Data files unchanged. See reports/public_scores.csv for previously confirmed submissions; this run does not submit to Kaggle.',flush=True)

if __name__=='__main__':
    main()

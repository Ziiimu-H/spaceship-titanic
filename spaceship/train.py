"""Train a baseline or CatBoost with group-aware validation by default."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import build_features
from .ablation import feature_variant, VARIANTS


def make_model(frame, args):
    categorical = frame.select_dtypes(include=['object','string']).columns.tolist()
    if args.model == 'catboost':
        return CatBoostClassifier(iterations=args.iterations,depth=args.depth,
            learning_rate=.04,l2_leaf_reg=args.l2,random_seed=42,thread_count=4,
            cat_features=categorical,loss_function='Logloss',verbose=False,allow_writing_files=False)
    numeric = [c for c in frame if c not in categorical]
    return Pipeline([
        ('preprocess',ColumnTransformer([
            ('numeric',SimpleImputer(strategy='median',add_indicator=True),numeric),
            ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical),
        ])),
        ('model',RandomForestClassifier(n_estimators=300,max_depth=14,
            min_samples_leaf=2,max_features=.8,random_state=42,n_jobs=4)),
    ])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    parser.add_argument('--output-dir',type=Path,default=Path('runs/train'))
    parser.add_argument('--model',choices=['random-forest','catboost'],default='catboost')
    parser.add_argument('--features',choices=['base',*VARIANTS],default='full')
    parser.add_argument('--split',choices=['group','stratified'],default='group')
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--split-seed',type=int,default=42)
    parser.add_argument('--iterations',type=int,default=900)
    parser.add_argument('--depth',type=int,default=6)
    parser.add_argument('--l2',type=float,default=5)
    args=parser.parse_args()
    if args.folds < 2:
        parser.error('--folds must be at least 2')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(args.data_dir/'train.csv',dtype={'PassengerId':str})
    test=pd.read_csv(args.data_dir/'test.csv',dtype={'PassengerId':str})
    sample=pd.read_csv(args.data_dir/'sample_submission.csv',dtype={'PassengerId':str})
    assert train.PassengerId.is_unique and test.PassengerId.is_unique
    assert test.PassengerId.equals(sample.PassengerId)
    assert train.Transported.notna().all()
    y=train.Transported.astype(int).to_numpy()
    roster=pd.concat([train.drop(columns='Transported'),test],ignore_index=True)
    if args.features == 'base':
        x,xt=build_features(train),build_features(test)
    else:
        x,xt=feature_variant(train,roster,args.features),feature_variant(test,roster,args.features)
    assert x.columns.tolist()==xt.columns.tolist()
    groups=train.PassengerId.str.split('_').str[0]
    splitter=(StratifiedGroupKFold if args.split=='group' else StratifiedKFold)(
        args.folds,shuffle=True,random_state=args.split_seed)
    split=splitter.split(x,y,groups) if args.split=='group' else splitter.split(x,y)
    probability=np.full(len(y),np.nan); fold_id=np.zeros(len(y),dtype=int); scores=[]
    for k,(tr,va) in enumerate(split,1):
        if args.split=='group':
            assert set(groups.iloc[tr]).isdisjoint(groups.iloc[va])
        model=make_model(x,args); model.fit(x.iloc[tr],y[tr])
        probability[va]=model.predict_proba(x.iloc[va])[:,1]; fold_id[va]=k
        score=float(((probability[va]>.5)==y[va]).mean())
        scores.append(dict(fold=k,accuracy=score,n_valid=len(va)))
        print(f'Fold {k}: {score:.4%}',flush=True)
    assert np.isfinite(probability).all()
    final=make_model(x,args); final.fit(x,y)
    test_probability=final.predict_proba(xt)[:,1]
    submission=pd.DataFrame({'PassengerId':test.PassengerId,'Transported':test_probability>.5})
    submission.to_csv(args.output_dir/'submission.csv',index=False)
    pd.testing.assert_frame_equal(submission,pd.read_csv(args.output_dir/'submission.csv',dtype={'PassengerId':str}))
    pd.DataFrame({'PassengerId':train.PassengerId,'Actual':y,'fold':fold_id,'probability':probability}).to_csv(args.output_dir/'oof.csv',index=False)
    pd.DataFrame(scores).to_csv(args.output_dir/'fold_scores.csv',index=False)
    metric={'model':args.model,'features':args.features,'split':args.split,'folds':args.folds,
        'split_seed':args.split_seed,'iterations':args.iterations if args.model=='catboost' else None,
        'depth':args.depth if args.model=='catboost' else None,'l2':args.l2 if args.model=='catboost' else None,
        'oof_accuracy':float(((probability>.5)==y).mean()),'threshold':.5,
        'train_test_shared_groups':len(set(groups)&set(test.PassengerId.str.split('_').str[0])),
        'submission_sha256':hashlib.sha256((args.output_dir/'submission.csv').read_bytes()).hexdigest(),
        'note':'Development CV. Enhanced features use label-free full-roster context. No Kaggle score measured by this script.'}
    (args.output_dir/'metrics.json').write_text(json.dumps(metric,indent=2)+'\n')
    print(json.dumps(metric,indent=2))

if __name__=='__main__':
    main()

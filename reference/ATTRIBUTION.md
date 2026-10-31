# Attribution and reproduction scope

The public four-member TabPFN probability files were authored by **TaeyanG4**
and are reused unchanged under **Creative Commons Attribution 4.0 International
(CC BY 4.0)**. SHA-256 values are recorded in `asset_hashes.json`.
Original notices and README are preserved under `reference/NOTICE.md` and
`reference/kaggle_dataset/README.md` relative to this directory.

- Probability dataset: https://www.kaggle.com/datasets/taeyangg4/spaceship-titanic-tabpfn-member-predictions
- Original source: https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments
- Reference commit: `332fd48f70a810432362c1e27c9bc00ae3554cec`
- License: https://creativecommons.org/licenses/by/4.0/
- Fusion notebook by **kalidenaiarys**: https://www.kaggle.com/code/kalidenaiarys/spaceship-titanic-tabpfn-catboost-lb-0-82534

Cached CatBoost and LightGBM probabilities and training reports are retained
from the fusion notebook as source evidence. Neither final submission uses the
cached LightGBM probabilities.

The local script independently implements the described logistic and spline
meta-model fitting. It does not train TabPFN, download pretrained weights,
or independently verify the source base-model training. No hidden competition
test labels are included.

The authors report public scores 0.82604 (four-member baseline) and 0.82534
(blend). The project account subsequently confirmed both values by actual
Kaggle submissions on 2026-10-31 (user-provided screenshot). The generated
CSVs match the source submission checksums. See reports/public_scores.csv.

The probability files retain CC BY 4.0. This notice does not grant rights over
third-party model weights or other third-party material.

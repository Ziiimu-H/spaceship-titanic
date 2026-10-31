"""独立消融第二版特征，不修改已提交版本。"""
from .features import build_features, SPENDING

VARIANTS = ['full', 'no_group_id', 'no_surname', 'no_group_fill', 'no_spend_features']


def feature_variant(frame, roster, variant='full'):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown feature variant: {variant}')
    result = build_features(frame, roster, enhanced=True)
    if variant == 'no_group_id':
        result = result.drop(columns=['GroupNumber', 'GroupPosition'])
    elif variant == 'no_surname':
        result = result.drop(columns='SurnameCount')
    elif variant == 'no_group_fill':
        original = build_features(frame)
        result[['HomePlanet', 'Deck', 'Side']] = original[['HomePlanet', 'Deck', 'Side']]
    elif variant == 'no_spend_features':
        result = result.drop(columns=[
            'KnownSpendingTotal', 'SpendingMissingCount', 'SpendState',
            'LuxurySpend', 'AmenitiesUsed', *[f'{col}Missing' for col in SPENDING],
        ])
        result['CryoSleep'] = build_features(frame)['CryoSleep']
    return result

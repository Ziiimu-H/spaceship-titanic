"""第二版特征：逐行变换 + 显式无标签乘客名册上下文。"""
import numpy as np
import pandas as pd

SPENDING = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
CATEGORIES = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']


def split_cabin(frame):
    return frame['Cabin'].astype('string').str.split('/', expand=True).reindex(columns=[0, 1, 2])


def build_features(frame, roster=None, enhanced=False):
    """不读取标签，不修改输入；roster 只能作为无标签名册使用。"""
    features = frame.drop(columns=['PassengerId', 'Name', 'Transported', 'Cabin'], errors='ignore').copy()
    cabin = split_cabin(frame)
    features['Deck'] = cabin[0]
    features['CabinNumber'] = pd.to_numeric(cabin[1], errors='coerce').astype(float)
    features['Side'] = cabin[2]
    features['KnownSpendingTotal'] = features[SPENDING].sum(axis=1, min_count=1)
    features['SpendingMissingCount'] = features[SPENDING].isna().sum(axis=1)

    if enhanced:
        if roster is None:
            raise ValueError('增强特征需要显式提供无标签 roster。')
        context = roster.drop(columns='Transported', errors='ignore').copy()
        if not context['PassengerId'].is_unique:
            raise ValueError('roster 中 PassengerId 必须唯一。')
        group = frame['PassengerId'].str.split('_').str[0]
        context_group = context['PassengerId'].str.split('_').str[0]
        group_counts = context_group.value_counts()
        features['GroupSize'] = group.map(group_counts).astype(float)
        features['IsAlone'] = (features['GroupSize'] == 1).astype(int)
        features['GroupNumber'] = pd.to_numeric(group, errors='coerce').astype(float)
        features['GroupPosition'] = pd.to_numeric(frame['PassengerId'].str.split('_').str[1], errors='coerce').astype(float)
        surname = frame['Name'].astype('string').str.split().str[-1]
        context_surname = context['Name'].astype('string').str.split().str[-1]
        features['SurnameCount'] = surname.map(context_surname.value_counts()).astype(float)
        features['CabinRegion'] = features['CabinNumber'] // 300
        # 同组属性一致时才补齐；出现冲突时不猜。
        context_cabin = split_cabin(context)
        for col, values in [('HomePlanet', context['HomePlanet']), ('Deck', context_cabin[0]), ('Side', context_cabin[2])]:
            known = pd.DataFrame({'group': context_group, 'value': values}).dropna()
            unique_counts = known.groupby('group')['value'].nunique()
            mapping = known.groupby('group')['value'].first().loc[unique_counts == 1]
            features[col] = features[col].fillna(group.map(mapping))
        # 未知或不完整的零消费不等同于已确认零消费，更不能据此强填休眠 True。
        total = features['KnownSpendingTotal']
        complete = features['SpendingMissingCount'] == 0
        features['SpendState'] = np.select(
            [total > 0, complete & (total == 0)], ['Positive', 'Zero'], default='Unknown'
        )
        features['LuxurySpend'] = features[['RoomService', 'Spa', 'VRDeck']].sum(axis=1, min_count=1)
        features['AmenitiesUsed'] = (features[SPENDING] > 0).sum(axis=1)
        for col in ['Age', *SPENDING]:
            features[f'{col}Missing'] = features[col].isna().astype(int)
        missing_sleep = features['CryoSleep'].isna()
        features.loc[missing_sleep & (total > 0), 'CryoSleep'] = False

    # 字符串类别缺失显式编码；数值 NaN 保留交给各模型的训练过程处理。
    for col in CATEGORIES:
        features[col] = features[col].astype('string').fillna('Unknown').astype(object)
    for col in features.columns.difference(CATEGORIES + ['SpendState']):
        features[col] = pd.to_numeric(features[col], errors='coerce').astype(float)
    return features

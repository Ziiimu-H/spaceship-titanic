import unittest
import pandas as pd
import numpy as np
from spaceship.features import build_features

class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'PassengerId': ['0001_01', '0001_02', '0002_01'],
            'Name': ['A Same', 'B Same', None],
            'HomePlanet': ['Earth', None, 'Mars'],
            'CryoSleep': [None, False, None],
            'Cabin': ['G/3/P', None, 'F/302/S'],
            'Destination': ['X', 'X', None], 'Age': [20., np.nan, 30.],
            'VIP': [False, None, True],
            **{c: [np.nan, 0., 1.] for c in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']},
        })

    def test_unknown_spending_is_not_zero(self):
        f = build_features(self.data, self.data, enhanced=True)
        self.assertTrue(pd.isna(f.loc[0, 'KnownSpendingTotal']))
        self.assertEqual(f.loc[0, 'SpendState'], 'Unknown')
        self.assertEqual(f.loc[1, 'SpendState'], 'Zero')
        self.assertEqual(f.loc[0, 'CryoSleep'], 'Unknown')
        self.assertEqual(f.loc[2, 'CryoSleep'], 'False')

    def test_group_context_and_missing_names(self):
        f = build_features(self.data, self.data, enhanced=True)
        self.assertEqual(f['GroupSize'].tolist(), [2, 2, 1])
        self.assertEqual(f.loc[1, 'HomePlanet'], 'Earth')
        self.assertEqual(f.loc[1, 'Deck'], 'G')
        self.assertEqual(f.loc[1, 'SurnameCount'], 2)
        self.assertTrue(pd.isna(f.loc[2, 'SurnameCount']))

    def test_conflicting_groups_are_not_imputed(self):
        extra = self.data.iloc[[0]].copy()
        extra['PassengerId'] = '0001_03'
        extra['HomePlanet'] = 'Europa'
        f = build_features(self.data, pd.concat([self.data, extra]), enhanced=True)
        self.assertEqual(f.loc[1, 'HomePlanet'], 'Unknown')

    def test_target_independence_and_no_input_mutation(self):
        original = self.data.copy(deep=True)
        a = self.data.assign(Transported=[True, False, True])
        b = self.data.assign(Transported=[False, True, False])
        pd.testing.assert_frame_equal(build_features(a, a, True), build_features(b, b, True))
        build_features(self.data, self.data, True)
        pd.testing.assert_frame_equal(original, self.data)

    def test_row_order_and_schema(self):
        rows = self.data.iloc[[2, 0]]
        f = build_features(rows, self.data, True)
        self.assertEqual(f.index.tolist(), [2, 0])
        self.assertNotIn('Transported', f)
        self.assertNotIn('PassengerId', f)
        self.assertNotIn('Name', f)
        self.assertEqual(f.loc[2, 'CabinRegion'], 1)
        self.assertEqual(f.loc[0, 'GroupSize'], 2)

if __name__ == '__main__':
    unittest.main()

import unittest
import pandas as pd
from spaceship.features import build_features
from spaceship.ablation import feature_variant


class AblationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            'PassengerId': ['0001_01','0001_02','0002_01'],
            'Name': ['A Same','B Same',None],
            'HomePlanet': ['Earth',None,'Mars'], 'CryoSleep':[None,False,None],
            'Cabin':['G/3/P',None,'F/302/S'], 'Destination':['X','X',None],
            'Age':[20.,float('nan'),30.], 'VIP':[False,None,True],
            'Transported':[True,False,True],
            **{c:[float('nan'),0.,1.] for c in ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']},
        })
        self.roster = self.frame.drop(columns='Transported').copy()
        self.full = build_features(self.frame, self.roster, True)

    def test_full_exactly_matches_v2(self):
        pd.testing.assert_frame_equal(self.full, feature_variant(self.frame, self.roster, 'full'))

    def test_remove_id_preserves_group_size(self):
        result = feature_variant(self.frame, self.roster, 'no_group_id')
        self.assertNotIn('GroupNumber', result)
        self.assertNotIn('GroupPosition', result)
        pd.testing.assert_series_equal(result['GroupSize'], self.full['GroupSize'])

    def test_no_surname(self):
        self.assertNotIn('SurnameCount', feature_variant(self.frame, self.roster, 'no_surname'))

    def test_no_group_fill_restores_original_categories(self):
        data = self.frame.copy()
        data.loc[data.index[0], ['HomePlanet', 'Cabin']] = None
        result = feature_variant(data, self.roster, 'no_group_fill')
        self.assertEqual(result.iloc[0]['HomePlanet'], 'Unknown')
        self.assertEqual(result.iloc[0]['Deck'], 'Unknown')
        self.assertEqual(result.iloc[0]['Side'], 'Unknown')

    def test_no_spending_engineering_retains_raw_spending(self):
        result = feature_variant(self.frame, self.roster, 'no_spend_features')
        self.assertNotIn('LuxurySpend', result)
        self.assertNotIn('KnownSpendingTotal', result)
        self.assertNotIn('SpendState', result)
        pd.testing.assert_series_equal(result['Spa'], self.full['Spa'])

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            feature_variant(self.frame, self.roster, 'typo')

if __name__ == '__main__':
    unittest.main()

import unittest
import polars as pl
from src.polars_scripts.static_transformers import CustomOneHotEncoding
from src.const import constants

class TestCustomOneHotEncoding(unittest.TestCase):
    def setUp(self):
        self. cu = CustomOneHotEncoding(
            single_val_cols=['A', 'B'],
            multi_val_cols=['C'],
            cumprob_inc_thresh=0.99,
            null_vals=constants.NULL_LIST,
            vocabthresh=100
        )

        self.dff = pl.DataFrame(
            {
                'A': ['a', 'b', 'null', None, 'z', 'aa'],
                'B': [12, 12, None, 1, 15, 2],
                'C': [[], ['a', 'c'], ['a'], ['z'], ['a', 'b', 'c'], None]
            }
        )

        self.dff_k = pl.DataFrame(
            {
                'A': ['a', 'b', 'null', None, 'zz', '1a'],
                'B': [12, 12, None, 1, 19, 2],
                'C': [[], ['a', 'c'], ['a'], ['k'], ['a', 'k', 'c'], None]
            }
        )

    def test_fit_transform(self):
        dff_ohe = self.cu.fit_transform(self.dff)
        expected_columns = ['A_null', 'A_unk', 'A_aa', 'A_a', 'A_z', 'A_b', 
                            'B_null', 'B_unk', 'B_1', 'B_2', 'B_12', 'B_15', 
                            'C_null', 'C_unk', 'C_a', 'C_b', 'C_z', 'C_c']
        self.assertEqual(set(dff_ohe.columns), set(expected_columns))
        # Add more checks for the content if needed

    def test_transform(self):
        self.cu.fit(self.dff)
        dff_k_ohe = self.cu.transform(self.dff_k)
        expected_columns = ['A_null', 'A_unk', 'A_aa', 'A_a', 'A_z', 'A_b', 
                            'B_null', 'B_unk', 'B_1', 'B_2', 'B_12', 'B_15', 
                            'C_null', 'C_unk', 'C_a', 'C_b', 'C_z', 'C_c']
        self.assertEqual(set(dff_k_ohe.columns), set(expected_columns))
        # Add more checks for the content if needed
    

if __name__ == "__main__":
    unittest.main()
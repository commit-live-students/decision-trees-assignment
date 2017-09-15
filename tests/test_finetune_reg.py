from unittest import TestCase
from sklearn.metrics import r2_score

class TestFinetune_reg(TestCase):
    def test_finetune_reg(self):
        from build import load_data, finetune_reg

        X_train, X_test, y_train, y_test = load_data('./data/energy_efficiency.csv', skiprows=1)

        param_grid = {"max_depth": [None, 6, 8, 10],
                      "max_leaf_nodes": [None, 5, 10, 20],
                      "min_impurity_split": [0.1, 0.2, 0.3]}

        y_pred, params = finetune_reg(X_train, X_test, y_train, param_grid)
        r2 = r2_score(y_pred, y_test)
        self.assertGreater(r2, 0.9)
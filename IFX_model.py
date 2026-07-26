import xgboost as xgb
import shap
import numpy as np
from sklearn.base import BaseEstimator

class IFX_XGBoost(BaseEstimator):
    """
    Iterative Feature eXclusion (IFX) wrapper for XGBoost.
    Handles binary, multi-class classification, and regression.
    """
    def __init__(self, n_iterations=5, early_stopping_rounds=10,
                 eval_metric=None, random_state=42, **xgb_params):
        self.n_iterations = n_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.xgb_params = xgb_params
        self.model = None
        self.features_removed = []
        self.feature_names = None
        self.num_class = None

    def get_params(self, deep=True):
        params = {
            'n_iterations': self.n_iterations,
            'early_stopping_rounds': self.early_stopping_rounds,
            'eval_metric': self.eval_metric,
            'random_state': self.random_state,
        }
        params.update(self.xgb_params)
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key in ['n_iterations', 'early_stopping_rounds', 'eval_metric', 'random_state']:
                setattr(self, key, value)
            else:
                self.xgb_params[key] = value
        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if X_val is not None:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)

        n_features = X_train.shape[1]
        self.feature_names = [f'f{i}' for i in range(n_features)]

        # ---------- Determine objective ----------
        # If the user provided an objective, respect it.
        if 'objective' in self.xgb_params:
            user_obj = self.xgb_params['objective']
            if user_obj == 'multi:softprob':
                # Multi‑class: set num_class if not provided
                unique_labels = np.unique(y_train)
                self.num_class = len(unique_labels)
                if 'num_class' not in self.xgb_params:
                    self.xgb_params['num_class'] = self.num_class
                if self.eval_metric is None:
                    self.eval_metric = 'mlogloss'
            elif user_obj == 'binary:logistic':
                if self.eval_metric is None:
                    self.eval_metric = 'logloss'
            else:
                # Regression (e.g., 'reg:squarederror')
                if self.eval_metric is None:
                    self.eval_metric = 'rmse'
        else:
            # No objective provided → auto‑detect
            unique_labels = np.unique(y_train)
            self.num_class = len(unique_labels)
            if self.num_class > 2:
                # Multi‑class classification
                self.xgb_params['objective'] = 'multi:softprob'
                self.xgb_params['num_class'] = self.num_class
                if self.eval_metric is None:
                    self.eval_metric = 'mlogloss'
            else:
                # Check if it's binary classification or regression
                # If labels are integer and only 2 values → binary classification
                if np.issubdtype(y_train.dtype, np.integer) and self.num_class == 2:
                    self.xgb_params['objective'] = 'binary:logistic'
                    if self.eval_metric is None:
                        self.eval_metric = 'logloss'
                else:
                    # Otherwise, treat as regression
                    self.xgb_params['objective'] = 'reg:squarederror'
                    if self.eval_metric is None:
                        self.eval_metric = 'rmse'

        # Ensure eval_metric is set
        if 'eval_metric' not in self.xgb_params:
            self.xgb_params['eval_metric'] = self.eval_metric

        # ---------- Initial training ----------
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals = [(dtrain, 'train'), (dval, 'val')]
        else:
            evals = [(dtrain, 'train')]

        params = self.xgb_params.copy()
        params['seed'] = self.random_state

        self.model = xgb.train(
            params, dtrain, num_boost_round=1000,
            evals=evals, early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False
        )

        # ---------- IFX iterations ----------
        active_features = np.ones(n_features, dtype=bool)

        for iteration in range(self.n_iterations):
            if np.sum(active_features) <= 1:
                break

            # Compute feature importance
            if X_val is not None:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_val)
                if isinstance(shap_values, list):
                    shap_vals = np.stack(shap_values, axis=-1)
                else:
                    shap_vals = shap_values
                if shap_vals.ndim == 3:
                    importance = np.mean(np.abs(shap_vals), axis=(0, 2))
                else:
                    importance = np.mean(np.abs(shap_vals), axis=0)
            else:
                imp_dict = self.model.get_score(importance_type='gain')
                if not imp_dict:
                    imp_dict = self.model.get_score(importance_type='weight')
                imp_array = np.zeros(n_features)
                for fname, imp in imp_dict.items():
                    if fname in self.feature_names:
                        idx = self.feature_names.index(fname)
                    else:
                        if fname.startswith('f'):
                            try:
                                idx = int(fname[1:])
                            except ValueError:
                                idx = -1
                        else:
                            idx = -1
                    if 0 <= idx < n_features:
                        imp_array[idx] = imp
                importance = imp_array

            # Ensure importance is 1D and length n_features
            if importance.ndim != 1 or len(importance) != n_features:
                if importance.ndim == 2 and importance.shape[0] == n_features:
                    importance = np.mean(importance, axis=1)
                else:
                    raise RuntimeError(f"Importance shape mismatch: expected ({n_features},), got {importance.shape}")

            importance[~active_features] = -np.inf
            top_feature = np.argmax(importance)
            if importance[top_feature] == -np.inf:
                break

            self.features_removed.append(top_feature)
            X_train[:, top_feature] = 0.0
            if X_val is not None:
                X_val[:, top_feature] = 0.0

            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
            if X_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
                evals = [(dtrain, 'train'), (dval, 'val')]
            else:
                evals = [(dtrain, 'train')]

            self.model = xgb.train(
                params, dtrain, num_boost_round=1000,
                evals=evals, early_stopping_rounds=self.early_stopping_rounds,
                xgb_model=self.model, verbose_eval=False
            )

            active_features[top_feature] = False

        return self

    def predict(self, X):
        X = np.asarray(X)
        X_pred = X.copy()
        for feat in self.features_removed:
            X_pred[:, feat] = 0.0
        dtest = xgb.DMatrix(X_pred, feature_names=self.feature_names)
        return self.model.predict(dtest)

    def predict_proba(self, X):
        return self.predict(X)
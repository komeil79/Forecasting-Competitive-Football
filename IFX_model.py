import xgboost as xgb
import shap
import numpy as np
from copy import deepcopy

class IFX_XGBoost:
    """
    Iterative Feature eXclusion (IFX) wrapper for XGBoost.
    
    Reimplementation :
    - Train an XGBoost model on all features.
    - Compute SHAP feature importance and find the most important feature.
    - Remove (zero out) that feature from the dataset.
    - Continue training the same model with early stopping.
    - Repeat until a set number of exclusions or until performance stops improving.
    """
    def __init__(self, n_iterations=5, early_stopping_rounds=10, 
                 eval_metric='mlogloss', random_state=42, **xgb_params):
        self.n_iterations = n_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.xgb_params = xgb_params
        self.model = None
        self.features_removed = []
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model with IFX.
        X_train, y_train : training data
        X_val, y_val     : validation data (for early stopping & SHAP)
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if X_val is not None:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)
        
        n_features = X_train.shape[1]
        # Prepare DMatrices
        dtrain = xgb.DMatrix(X_train, label=y_train)
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'val')]
        else:
            evals = [(dtrain, 'train')]
        
        # Initial XGBoost parameters
        params = self.xgb_params.copy()
        params.setdefault('objective', 'binary:logistic')
        params.setdefault('eval_metric', self.eval_metric)
        params.setdefault('seed', self.random_state)
        
        # Train initial model
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False
        )
        
        # Track the current feature mask (True = feature still active)
        active_features = np.ones(n_features, dtype=bool)
        
        for iteration in range(self.n_iterations):
            # If no features left, break
            if np.sum(active_features) <= 1:
                break
            
            # Compute SHAP values on validation set (if available, else training)
            if X_val is not None:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_val)
                # Average absolute SHAP value per feature
                importance = np.mean(np.abs(shap_values), axis=0)
            else:
                # Use built-in gain importance if no validation
                importance = self.model.get_score(importance_type='gain')
                # map feature names to indices (if feature names are given)
                # We'll assume feature names are 'f0', 'f1', ... if not set
                if len(importance) == 0:
                    # fallback: use number of splits
                    importance = self.model.get_score(importance_type='weight')
                # Create array of importance for all features
                imp_array = np.zeros(n_features)
                for fname, imp in importance.items():
                    if fname.startswith('f'):
                        idx = int(fname[1:])
                        if idx < n_features:
                            imp_array[idx] = imp
                importance = imp_array
            
            # Mask out inactive features (set their importance to -inf)
            importance[~active_features] = -np.inf
            # Find most important active feature
            top_feature = np.argmax(importance)
            if importance[top_feature] == -np.inf:
                break  # no active feature left
            
            self.features_removed.append(top_feature)
            # Remove (zero out) this feature in the data
            X_train[:, top_feature] = 0.0
            if X_val is not None:
                X_val[:, top_feature] = 0.0
            
            # Update DMatrices
            dtrain = xgb.DMatrix(X_train, label=y_train)
            if X_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val)
                evals = [(dtrain, 'train'), (dval, 'val')]
            else:
                evals = [(dtrain, 'train')]
            
            # Continue training the same model
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=evals,
                early_stopping_rounds=self.early_stopping_rounds,
                xgb_model=self.model,
                verbose_eval=False
            )
            
            # Mark feature as inactive
            active_features[top_feature] = False
        
        return self
    
    def predict(self, X):
        """Predict probabilities."""
        X = np.asarray(X)
        # Ensure removed features are zeroed out during prediction
        X_pred = X.copy()
        for feat in self.features_removed:
            X_pred[:, feat] = 0.0
        dtest = xgb.DMatrix(X_pred)
        return self.model.predict(dtest)
    
    def predict_proba(self, X):
        """Alias for predict."""
        return self.predict(X)
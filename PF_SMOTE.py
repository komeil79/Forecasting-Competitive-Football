import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

class PF_SMOTE:
    """
    Parameter-Free SMOTE (PF-SMOTE).
    
    Characteristics:
    - Defines boundary minority examples (those with majority neighbours) and safe minority examples.
    - For safe examples: generates synthetic points by interpolation between two safe minority points.
    - For boundary examples: generates synthetic points by adding Gaussian noise to expand the margin.
    """
    def __init__(self, k_neighbors=5, random_state=None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        """
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) with binary labels (0=majority, 1=minority)
        Returns (X_resampled, y_resampled) with synthetic minority added.
        """
        rng = check_random_state(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Identify minority and majority
        minority_label = 1
        majority_label = 0
        X_min = X[y == minority_label]
        X_maj = X[y == majority_label]
        n_min = len(X_min)
        n_maj = len(X_maj)
        
        if n_min == 0 or n_maj == 0:
            raise ValueError("Both classes must be present.")
        
        # Determine effective k (parameter‑free: use sqrt(n_min) if not specified)
        if self.k_neighbors is None:
            k = int(np.sqrt(n_min))
        else:
            k = min(self.k_neighbors, n_min - 1)
        
        # Find nearest neighbours for each minority point (among all points)
        nn = NearestNeighbors(n_neighbors=k+1)  # +1 to exclude itself
        all_data = np.vstack([X_min, X_maj])
        nn.fit(all_data)
        distances, indices = nn.kneighbors(X_min, n_neighbors=k+1)
        # indices: row i -> neighbours of minority i (including itself as first)
        # We need to count how many of the k neighbours (excluding self) are majority
        # All points: first n_min are minority, rest are majority
        majority_indices = np.arange(n_min, n_min + n_maj)
        is_boundary = []
        for i in range(n_min):
            neigh_indices = indices[i, 1:]  # exclude self
            majority_count = np.sum(np.isin(neigh_indices, majority_indices))
            if majority_count > 0:
                is_boundary.append(True)
            else:
                is_boundary.append(False)
        is_boundary = np.array(is_boundary)
        
        # Separate boundary and safe minority
        X_boundary = X_min[is_boundary]
        X_safe = X_min[~is_boundary]
        
        # Number of synthetic examples to generate (to balance classes)
        n_synthetic = n_maj - n_min
        if n_synthetic <= 0:
            return X, y  # already balanced or majority is smaller
        
        synthetic = []
        
        # --- Generate from safe examples ---
        # We'll interpolate between two randomly chosen safe points
        if len(X_safe) >= 2:
            n_safe_gen = int(n_synthetic * 0.5)  # half from safe, half from boundary (adjustable)
            for _ in range(n_safe_gen):
                idx1, idx2 = rng.choice(len(X_safe), 2, replace=False)
                p1 = X_safe[idx1]
                p2 = X_safe[idx2]
                # convex combination with random lambda in (0,1)
                lam = rng.rand()
                new_point = lam * p1 + (1 - lam) * p2
                synthetic.append(new_point)
        else:
            n_safe_gen = 0
        
        # --- Generate from boundary examples using Gaussian noise ---
        # For each boundary point, we compute the average distance to its minority neighbours
        # (as a proxy for local density) and add Gaussian noise scaled by that distance.
        if len(X_boundary) > 0:
            # We need distances to minority neighbours for boundary points
            # Recompute nearest neighbours among minority only
            if len(X_min) >= 2:
                nn_min = NearestNeighbors(n_neighbors=min(k, len(X_min)-1))
                nn_min.fit(X_min)
                dist_min, _ = nn_min.kneighbors(X_boundary)
                # average distance per boundary point
                avg_dist = np.mean(dist_min, axis=1)
            else:
                avg_dist = np.ones(len(X_boundary)) * 0.1  # fallback
            
            n_boundary_gen = n_synthetic - n_safe_gen
            for i in range(n_boundary_gen):
                # pick a random boundary point
                idx = rng.randint(0, len(X_boundary))
                point = X_boundary[idx]
                std = avg_dist[idx] if avg_dist is not None else 0.1
                noise = rng.normal(0, std, size=point.shape)
                new_point = point + noise
                synthetic.append(new_point)
        
        synthetic = np.array(synthetic)
        if len(synthetic) == 0:
            return X, y
        
        X_resampled = np.vstack([X, synthetic])
        y_resampled = np.hstack([y, np.ones(len(synthetic), dtype=y.dtype)])
        return X_resampled, y_resampled
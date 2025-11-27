import numpy as np
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors

def mirror_neighbor_aug_with_shape(df, target_col, minority_class=1, total_synthetic=None,
                                   n_neighbors=5, shape_scale_range=(0.8, 1.2)):

    features = [c for c in df.columns if c not in [target_col, "synthetic_id"]]
    minority_df = df[df[target_col] == minority_class]
    X = minority_df[features].values

    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)

    synthetic_samples = []
    total_needed = total_synthetic or len(minority_df)

    for i in range(total_needed):
        idx = random.randint(0, len(minority_df)-1)
        original = X[idx]

        _, indices = nbrs.kneighbors([original])
        neighbor = X[random.choice(indices[0][1:])]

        direction = neighbor - original
        direction *= np.random.uniform(shape_scale_range[0],
                                       shape_scale_range[1],
                                       size=len(direction))

        scale = np.random.uniform(0.6, 1.4)
        synthetic = original + scale * direction

        synthetic_samples.append(dict(
            {target_col: minority_class},
            **dict(zip(features, synthetic))
        ))

    return pd.DataFrame(synthetic_samples)

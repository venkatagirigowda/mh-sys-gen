import numpy as np
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors

def sequential_shadow_mirror(df, target_col, minority_class, ratio=1.0):

    features = [c for c in df.columns if c != target_col]

    minority_df = df[df[target_col] == minority_class]
    if minority_df.empty:
        raise ValueError(f"Minority class {minority_class} not present")

    total_synth = int(len(minority_df) * ratio)
    if total_synth < 1:
        raise ValueError("Ratio too small")

    X = minority_df[features].values

    # Phase 1: Shadow projection
    shadows = []
    for i in range(total_synth):
        idx = np.random.randint(0, len(X))
        original = X[idx]
        shadow = original * np.random.uniform(0.6, 1.4, size=len(features))
        shadows.append(shadow)

    shadows = np.array(shadows)

    # Phase 2: Mirror nearest neighbor reflection
    nbrs = NearestNeighbors(n_neighbors=2).fit(shadows)

    mirrored = []
    for i in range(total_synth):
        _, idxs = nbrs.kneighbors([shadows[i]])
        neighbor = shadows[idxs[0][1]]
        reflected = shadows[i] + (neighbor - shadows[i]) * np.random.uniform(0.5, 1.3)
        mirrored.append(reflected)

    synthetic_df = pd.DataFrame(mirrored, columns=features)
    synthetic_df[target_col] = minority_class

    combined = pd.concat([df, synthetic_df], ignore_index=True)

    X_out = combined.drop(columns=[target_col])
    y_out = combined[target_col]

    return X_out, y_out

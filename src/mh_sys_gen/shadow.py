import numpy as np
import pandas as pd

def shadow_augmentation(df, target_col, minority_class=1, total_synthetic=None,
                        light_dir=None, shadow_plane_normal=None, shadow_plane_point=None):

    features = [c for c in df.columns if c not in [target_col, "synthetic_id"]]
    minority_df = df[df[target_col] == minority_class]
    X = minority_df[features].values

    # Default light direction
    if light_dir is None:
        light_dir = np.array([1] + [0]*(len(features)-1))
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Default plane normal = light direction
    if shadow_plane_normal is None:
        shadow_plane_normal = light_dir
    shadow_plane_normal = shadow_plane_normal / np.linalg.norm(shadow_plane_normal)

    # Default plane origin = 0 vector
    if shadow_plane_point is None:
        shadow_plane_point = np.zeros(len(features))

    synthetic_samples = []
    total_needed = total_synthetic or len(minority_df)

    def project_point(point):
        w = point - shadow_plane_point
        dist = np.dot(w, shadow_plane_normal)
        return point - dist * shadow_plane_normal

    for i in range(total_needed):
        idx = np.random.randint(0, len(minority_df))
        original = X[idx]

        shadow = project_point(original)
        scale = np.random.uniform(0.5, 1.5)
        synthetic = shadow_plane_point + (shadow - shadow_plane_point) * scale

        synthetic_samples.append(dict(
            {target_col: minority_class},
            **dict(zip(features, synthetic))
        ))

    return pd.DataFrame(synthetic_samples)

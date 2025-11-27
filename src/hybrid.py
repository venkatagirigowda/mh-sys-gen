import pandas as pd
from .mirror import mirror_neighbor_aug_with_shape
from .shadow import shadow_augmentation

def hybrid_augmentation(df, target_col, minority_class=1, ratio=1.0):

    n = int(len(df[df[target_col] == minority_class]) * ratio)

    df_mirror = mirror_neighbor_aug_with_shape(df, target_col, minority_class, total_synthetic=n)
    df_shadow = shadow_augmentation(df, target_col, minority_class, total_synthetic=n)

    return pd.concat([df_mirror, df_shadow], ignore_index=True)

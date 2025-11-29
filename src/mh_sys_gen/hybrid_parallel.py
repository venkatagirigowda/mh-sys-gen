import pandas as pd
from .mirror import mirror_neighbor_aug_with_shape
from .shadow import shadow_augmentation

def hybrid_augmentation(df, target_col, minority_class, ratio=1.0):

    minority_df = df[df[target_col] == minority_class]

    if minority_df.empty:
        raise ValueError(f"Minority class '{minority_class}' not found in dataset.")


    total_synthetic = int(len(minority_df) * ratio)

    if total_synthetic < 1:
        raise ValueError("Ratio too small — produces zero synthetic samples.")

 
    df_mirror = mirror_neighbor_aug_with_shape(
        df=df,
        target_col=target_col,
        minority_class=minority_class,
        total_synthetic=total_synthetic
    )


    df_shadow = shadow_augmentation(
        df=df,
        target_col=target_col,
        minority_class=minority_class,
        total_synthetic=total_synthetic
    )


    synthetic_df = pd.concat([df_mirror, df_shadow], ignore_index=True)
    combined_df = pd.concat([df, synthetic_df], ignore_index=True)

 
    X = combined_df.drop(columns=[target_col, "synthetic_id"], errors="ignore")
    y = combined_df[target_col]

    return X, y

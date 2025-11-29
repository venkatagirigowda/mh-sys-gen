class MHSysGen:
    def __init__(self, method="parallel", ratio=1.0, minority_class=1):
        self.method = method
        self.ratio = ratio
        self.minority_class = minority_class

    def fit_resample(self, df, target):
        if self.method == "parallel":
            from .hybrid_parallel import hybrid_augmentation
            return hybrid_augmentation(df, target, self.minority_class, self.ratio)

        elif self.method == "sequential":
            from .hybrid_sequential import sequential_shadow_mirror
            return sequential_shadow_mirror(df, target, self.minority_class, self.ratio)

        else:
            raise ValueError("Invalid method. Use 'parallel' or 'sequential'.")

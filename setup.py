from setuptools import setup, find_packages

setup(
    name="mh-sys-gen",
    version="0.1.0",
    description="Mirror+Shadow hybrid augmentation for imbalanced tabular datasets",
    author="Venkatagiri Gowda",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn"
    ],
    license="MIT",
)

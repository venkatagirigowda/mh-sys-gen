from setuptools import setup, find_packages

setup(
    name="mh_sys_gen",
    version="0.3.3",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy"
    ],
    python_requires=">=3.8",
)

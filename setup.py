from setuptools import setup, find_packages

setup(
    name="ContextScore",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ],
    entry_points={
        "console_scripts": [
            "annotate-svs=scripts.annotate_svs:main",
            "train-model=scripts.train_model:main",
            "predict=scripts.predict:main",
        ]
    },
)

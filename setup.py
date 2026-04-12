from setuptools import setup, find_packages

setup(
    name="ContextScore",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
    ],
    extras_require={
        "plot": [
            "matplotlib",
            "seaborn",
        ]
    },
    entry_points={
        "console_scripts": [
            "contextscore=contextscore.predict:main",
        ]
    },
)

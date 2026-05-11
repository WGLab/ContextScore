from pathlib import Path
from setuptools import setup, find_packages


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILES = [
    path.relative_to(PROJECT_ROOT).as_posix()
    for path in (PROJECT_ROOT / "data").glob("*")
    if path.is_file()
]

setup(
    name="ContextScore",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    data_files=[("contextscore/data", DATA_FILES)],
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

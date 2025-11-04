from setuptools import setup,find_packages

setup(
    name="meatools",
    version="0.3.4",
    author='Jiangyuan John Feng, Shicheng Xu',
    description='',
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "cantera",
        "matplotlib",
        "scipy",
        "scikit-learn"
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mea = meatools.cli:main"
        ]
    }
)
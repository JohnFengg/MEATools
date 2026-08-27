from setuptools import setup,find_packages

setup(
   name="meatools",
    version="1.1.6",
   author='Jiangyuan John Feng, Shicheng Xu',
    description='',
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "cantera",
        "matplotlib",
        "scipy",
        "scikit-learn",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mea = meatools.cli:main"
        ]
    }
)

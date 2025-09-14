from setuptools import setup,find_packages

setup(
    name="meatools",
    version="0.1.0",
    author='Jiangyuan John Feng, Shicheng Xu',
    description='',
    requires=[
        "python",
        "numpy",
        "cantera",
        "matplotlib",
        "scipy"
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mea = meatools.cli:main"
        ]
    }
)
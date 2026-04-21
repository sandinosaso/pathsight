from setuptools import setup, find_packages

setup(
    name="pathsight-model-service",
    version="0.1.0",
    # This tells Python: "The code lives in model/src"
    package_dir={"": "model/src"},
    # This tells Python: "Find all folders with __init__.py inside model/src"
    packages=find_packages(where="model/src"),
)

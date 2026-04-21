from setuptools import setup, find_packages

setup(
    name="model-service",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.13",
        "numpy>=1.24",
        "python-dotenv>=1.0",
    ],
)

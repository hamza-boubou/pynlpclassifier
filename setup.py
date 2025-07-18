from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="PyNLPclassifier",
    version="1.0.2",
    author="Hamza Dev",
    author_email="you@example.com",
    description="A PySpark-based NLP pipeline for spelling correction and text classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hamza-boubou/pynlpclassifier",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pyspark>=3.0.0",
        "pandas>=1.0.0",
        "gensim>=3.7.0",
        "numpy>=1.18.0",
        "python-Levenshtein>=0.12.0",
        "py4j==0.10.9.7"
    ],
    license="MIT",
)

from setuptools import setup, find_packages

setup(
    name="orbit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "pyarrow>=6.0.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "transformers>=4.18.0",
        "datasets>=2.0.0",
        "torch>=1.10.0",
        "fasttext>=0.9.2",
    ],
    author="ORBIT Team",
    author_email="orbit@example.com",
    description="Domain-Specific AI for Astronomy, Law, and Medicine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ORBIT-Llama",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cross-entropy-method",
    version="0.1.0",
    license='MIT',
    author="Ido Greenberg",
    description="The Cross-Entropy Method for either rare-event sampling or optimization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url="https://github.com/ido90/CEM",
    keywords = ["cross entropy", "CEM", "sampling", "optimization"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.5',
    py_modules=["cross_entropy_method"],
    install_requires = ["numpy","scipy","pandas"]
)

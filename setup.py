from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='QAOAHP',
    version='0.1',
    description='A hybrid quantum computation platform',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Amir Alizadeh',
    author_email='N1259534@my.nty.ac.uk',
    url='https://github.com/yourusername/QAOAHP',  # update as needed
    packages=find_packages(where=".", exclude=["tests*"]),
    install_requires=requirements,
    python_requires=">=3.10",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)

"""
**Explanation of the setup.py improvements:**

1. **long_description and long_description_content_type**:  
   These fields allow your package to display a nicely formatted README on PyPI, making it more user-friendly and informative for potential users.

2. **Encoding for README.md**:  
   Using `encoding="utf-8"` ensures compatibility with non-ASCII characters in your README, preventing encoding errors.

3. **find_packages(where=".", exclude=["tests*"])**:  
   This ensures only your actual package code is included, not test directories, keeping your distribution clean.


5. **classifiers**:  
   These provide metadata for PyPI, making your package easier to find and categorize (e.g., by Python version, license, intended audience).


6. **include_package_data=True**:  
   Ensures non-Python files (like data files or resources) included in your package are also installed.

7. **entry_points**:  
   This is used for command-line tools. If you have a CLI, you can define it here; otherwise, it can be omitted.

"""
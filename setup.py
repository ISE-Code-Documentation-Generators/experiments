import setuptools
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "0.0.0.1"
DESCRIPTION = "To be added in the future"


setuptools.setup(
    name="ise_cdg_experiments",
    version=VERSION,
    author="Ashkan Khademian",
    author_email="ashkan.khd.q@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "git+https://github.com/ISE-Code-Documentation-Generators/models.git",
        "git+https://github.com/ISE-Code-Documentation-Generators/utility.git",
        "git+https://github.com/ISE-Code-Documentation-Generators/data.git",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
)

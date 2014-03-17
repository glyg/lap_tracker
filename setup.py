# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import lap_tracker

install_requires = ['cython', 'numpy', 'scipy', 'pandas', 'matplotlib']


def install_requirements(install_requires):
    """
    Install third party libs in right order.
    """
    import subprocess
    import pip

    for package in install_requires:
        try:
            __import__(package)
        except:
            pip.main(['install', package])

install_requirements(install_requires)

setup(
    name='lap_tracker',
    version=lap_tracker.__version__,
    packages=find_packages(),
    author="BNOI Project",
    author_email="bnoi.project@gmail.com",
    description="""Python implementation of the particle tracking alogorithm
                   described in K. Jaqaman and G. Danuser, Nature Methods, 2008.
                   See https://github.com/bnoi/lap_tracker for details.""",
    long_description=open('README.md').read(),
    install_requires=install_requires,
    include_package_data=True,
    url='https://github.com/bnoi/lap_tracker',
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    entry_points={
        'console_scripts': [
            #'proclame-sm = sm_lib.core:proclamer',
        ],
    },
)

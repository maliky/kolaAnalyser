# -*- coding: utf-8 -*-
"""
MarketAnalyser
"""
from setuptools import setup, find_packages

with open("README.rst", "r") as f:
    long_description = f.read()

setup(
    name="kolaAnalyser",
    version="0.1.0",
    description="A market analyser.  Transforming holc in homemade pipes and taking statistiques",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Malik Koné",
    author_email="malikykone@gmail.com",
    url="https://github.com/maliky/kolaAnalyser",
    packages=find_packages(),
    zip_safe=False,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "kolaAnalyse=kolaAnalyser.analyser:main_prg",
        ]
    },
    # la verions de websocket est importante
    install_requires=["pandas", "mlkHelper"],
    extras_require={
        "dev": ["mypy", "flake8", "black"],
        "packaging": ["twine"],
        "test": ["pytest", "hypothesis"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Natural Language :: French",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Topic :: Office/Business :: Financial",
        "Topic :: Utilities",
        "Topic :: System :: Monitoring",
    ],
    # package_data={"Doc": ["Doc/*"]},
)

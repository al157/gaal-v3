from setuptools import setup, find_packages

setup(
    name="gaal_v3",
    version="3.0.0",
    description="GAAL v3 — LangGraph-inspired StateGraph Arena Framework",
    packages=find_packages(include=["gaal_v3*"]),
    python_requires=">=3.11",
    install_requires=[
        "jinja2>=3.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "test": ["pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "gaal-arena=gaal_v3.run:main",
        ],
    },
)

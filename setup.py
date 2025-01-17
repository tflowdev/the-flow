"""The Flow: Data-centric declarative deep learning framework."""
from codecs import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README.md file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line]

extra_requirements = {}

with open(path.join(here, "requirements_serve.txt"), encoding="utf-8") as f:
    extra_requirements["serve"] = [line.strip() for line in f if line]

with open(path.join(here, "requirements_viz.txt"), encoding="utf-8") as f:
    extra_requirements["viz"] = [line.strip() for line in f if line]

with open(path.join(here, "requirements_distributed.txt"), encoding="utf-8") as f:
    extra_requirements["distributed"] = [line.strip() for line in f if line]

with open(path.join(here, "requirements_hyperopt.txt"), encoding="utf-8") as f:
    extra_requirements["hyperopt"] = [line.strip() for line in f if line]

with open(path.join(here, "requirements_tree.txt"), encoding="utf-8") as f:
    extra_requirements["tree"] = [line.strip() for line in f if line]

with open(path.join(here, "requirements_llm.txt"), encoding="utf-8") as f:
    extra_requirements["llm"] = [line.strip() for line in f if line]

with open(path.join(here, "requirements_explain.txt"), encoding="utf-8") as f:
    extra_requirements["explain"] = [line.strip() for line in f if line]

with open(path.join(here, "requirements_benchmarking.txt"), encoding="utf-8") as f:
    extra_requirements["benchmarking"] = [line.strip() for line in f if line]

extra_requirements["full"] = [item for sublist in extra_requirements.values() for item in sublist]

with open(path.join(here, "requirements_test.txt"), encoding="utf-8") as f:
    extra_requirements["test"] = extra_requirements["full"] + [line.strip() for line in f if line]

with open(path.join(here, "requirements_extra.txt"), encoding="utf-8") as f:
    extra_requirements["extra"] = [line.strip() for line in f if line]

setup(
    name="the-flow",
    version="0.1.0",
    description="AI Framework for Solana: Build, deploy, and scale AI models that seamlessly integrate with blockchain applications.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tflowdev/the-flow",
    download_url="https://pypi.org/project/the-flow/",
    author="The Flow",
    author_email="team@tflow.dev",
    license="Apache 2.0",
    keywords="solana blockchain ai machine_learning deep_learning defi analytics real_time transaction_processing",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    python_requires=">=3.8",
    include_package_data=True,
    package_data={"theflow": ["etc/*", "examples/*.py"]},
    install_requires=requirements,
    extras_require=extra_requirements,
    entry_points={"console_scripts": ["theflow=theflow.cli:main"]},
)

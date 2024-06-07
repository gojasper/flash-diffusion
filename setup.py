from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="flash",
    version="0.1",
    author="Clement Chadebec",
    author_email="clement.chadebec@jasper.ai",
    description="Flash Diffusion paper implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gojasper/flash-diffusion",
    project_urls={"Bug Tracker": "https://github.com/gojasper/flash-diffusion/issues"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "black>=24.2.0",
        "einops==0.7.0",
        "isort>=5.13.2",
        "lightning==2.2.5",
        "opencv-python==4.9.0.80",
        "pydantic>=2.6.1",
        "scipy>=1.12.0",
        "sentencepiece>=0.2.0",
        "tokenizers>=0.15.2",
        "transformers==4.38.0",
        "wandb==0.16.2",
        "webdataset>=0.2.86",
        "peft==0.9.0",
        "timm==0.9.16",
        "controlnet-aux==0.0.7",
        "lpips==0.1.4",
    ],
    python_requires=">=3.10",
)

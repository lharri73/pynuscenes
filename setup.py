
with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="nuscenes_dataset",
    version="0.1",
    author="Ramin Nabati, Landon Harris",
    description="A devkit for the NuScenes dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrnabati/nuscenes_dataset",
    packages=setuptools.find_packages("pynuscenes", exclude=['__pycache__', '*.__pycache__', '__pycache.*', '*.__pycache__.*']),
    package_dir={'':'pynuscenes'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires='>=3.6',
)
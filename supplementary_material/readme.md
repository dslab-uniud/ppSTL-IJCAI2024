# Setup and Installation

This project requires to have installed a few packages. Some of these packages need to be installed via conda, while others can be installed via pip.
Our suggestion is to create a new conda environment containing all the needed packages.

First, navigate to the project directory:

```bash
cd <project_directory>
```

From [requirements.txt](https://github.com/dslab-uniud/ppSTL-IJCAI2024/blob/main/supplementary_material/requirements.txt), prepare a file with packages available in conda and another for those only available in pip:

```bash
grep -v '=pypi_0' requirements.txt > conda_requirements.txt
grep '=pypi_0' requirements.txt | sed 's/=pypi_0//' > pip_requirements.txt
sed -i 's/=\+/==/g' pip_requirements.txt
```

To install the required packages, first use conda (creating a new local environment within your `<project_directory>`). Note that conda-forge channel might be required:

```bash
conda config --append channels conda-forge
conda create --prefix ./env --file conda_requirements.txt
```

Then, use pip for the remaining packages:

```bash
conda activate ./env
pip install -r pip_requirements.txt
```

Now, you should be able to run the scripts as described in the following "How to Execute the Code" section.

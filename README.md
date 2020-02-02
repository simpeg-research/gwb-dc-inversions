**[usage](#usage) | [running the notebooks](#running-the-notebooks) | [issues](#issues) | [citation](#citation) | [license](#license)**

# gwb-dc-inversions
Examples of 1D and 2D inversions for Geoscience Without Border Projects at Myanmar

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/simpeg-research/gwb-dc-inversions/master?filepath=notebooks%2Findex.ipynb)
[![Build Status](https://travis-ci.org/simpeg-research/gwb-dc-inversions.svg?branch=master)](https://travis-ci.org/simpeg-research/gwb-dc-inversions)

## Usage

### online
You can run these notebooks online through mybinder by clicking on the badge below:

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/simpeg-research/gwb-dc-inversions/master?filepath=notebooks%2Findex.ipynb)

### locally
To run the notebooks locally, you will need to have python installed,
preferably through [anaconda](https://www.anaconda.com/download/). Please download 
Python 3.7 or greater. 

Once you have downloaded and installed anaconda, you can then clone this repository. 
From a command line (if you are on windows, please use the anaconda terminal that came with the installation)
run

```
git clone https://github.com/simpeg-research/gwb-dc-inversions.git
```

Then `cd` into the `gwb-dc` directory:

```
cd gwb-dc-inversions
```

To setup your software environment, we recommend you use the provided conda environment

```
conda env create -f environment.yml
conda activate gwb-dc
```

You can then launch Jupyter

```
jupyter notebook
```

Jupyter will then launch in your web-browser.

## Running the notebooks

Each cell of code can be run with `shift + enter` or you can run the entire notebook by selecting `cell`, `Run All` in the toolbar.

<img src="https://em.geosci.xyz/_images/run_all_cells.png" width=80% align="middle">

For more information on running Jupyter notebooks, see the [Jupyter Documentation](https://jupyter.readthedocs.io/en/latest/)

If you are new to Python, I highly recommend taking a look at:
- [A Whirlwind Tour of Python](https://jakevdp.github.io/WhirlwindTourOfPython/)
- [The Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

## Issues

Please [make an issue](https://github.com/simpeg-research/gwb-dc-inversions/issues) if you encounter any problems while trying to run the notebooks.

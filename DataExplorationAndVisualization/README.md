# Data Exploration And Visualization
A tutorial on data exploration and visualization creating for the 2024 APS March Meeting.

## Summary
This tutorial is an introduction to data exploration and visualization using four Python libraries: [Pandas](https://pandas.pydata.org), [Seaborn](https://seaborn.pydata.org/index.html), [Matplotlib](https://matplotlib.org), and [Scikit-Learn](https://scikit-learn.org/stable/). Pandas is used for import a data file, as well as data exploration and manipulation. Seaborn and Matplotlib are used for data visualization. Scikit-Learn is used for modelling the data with simple machine learning models.

The dataset used in this tutorial contains information about the nucleon numbers, binding energies, and masses from all nuclei included in the [AME2016 atomic mass evaluation])(https://www-nds.iaea.org/amdc/ame2016/AME2016-a.pdf), however, prior knowledge of nuclear physics is not neccessary to complete this tutorial.

## Instructions for Running with Google Colab
[Google Colab](https://colab.research.google.com) is a cloud based software which runs Python notebooks. One of the main perks of Google Colab is that many popular Python libraries come pre-installed, so there is not set-up needed. You can run this tutorial on Google Colab by clicking the "Run in Colab" button at the top of the notebook and then clicking "Copy to Drive" on the toolbar of the Colab window which will open. This will give yoou a copy of the notebook in your Google Drive account which will save any changes you make. If you do not click "Copy to Drive" you will be running the code in a "playground" version of Google Colab which will allow you to change and run the notebook but will not save your changes.

## Instructions for Running Locally
To run this tutorial locally, you must have Python3 installed along with the following packages:

* Jupyter Notebooks
* Numpy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn

These libraries can be installed via pip, conda, brew, or any other package manager. Jupyter notebook can be used to run this tutorial. For information on running a notebook with Jupyter, refer to [Jupyter's tutorials](https://docs.jupyter.org/en/latest/). Note that the data file used in this notebook is imported directly from GitHub each time the notebook is run and is not the file stored locally. If you wish to change this you will need to modify the second code cell that initially creates the Pandas Dataframe.
# Welcome to pyMAGIQ

pyMAGIQ is a python-based MAGnetotelluric Impedance Qualifier trained by the USArray data.

## Requirements
Before running pyMAGIQ, please install the following packages. I use anaconda3-5.3.0 as a python environments. I recommend to use **pyenv** and **virtualenv** to have virtual environments.

## Examples
Example notebooks can be found in [notebooks/](https://github.com/nimamura/pyMAGIQ/tree/master/notebooks).
Example scripts for command line use can be found in [scripts/](https://github.com/nimamura/pyMAGIQ/tree/master/scripts).

## Install
1. Clone the git repository
```
git clone https://github.com/nimamura/pyMAGIQ.git
```
2. Install the pre-required software. Followings are in case of anaconda.
```
conda install tensorflow
conda install keras
conda install -c conda-forge cartopy
```
3. Build and install the package
```
python setup.py build
python setup.py install
```
4. Create survey directory. This directory has to include sub-directory as follows. Please put the Earthscope impedance tensors in the train directory.
```
outputs
preprocessed
train
unrated
```
Hierarchy of this directory is like the following.
```
pyMAGIQ
└── survey
    ├── outputs
    ├── preprocessed
    ├── train
    │   ├── MT_TF_USArray.ALW48.2015
    │   │   ├── ALW48bc_V47coh.png
    │   │   │── ALW48bc_V47coh.zrr
    │   │   │── USArray.ALW48.2015.edi
    │   │   │── USArray.ALW48.2015.xml
    │   ├── ...
    ├── unrated
    │   ├── MT_TF_CAFE-MT.CAF02.2010
    │   │   ├── CAF002.png
    │   │   │── ...
```

## Run code
1. Please run pre-process by using notebooks/preprocess.ipynb. Once pre-process has done, you will get csv files in preprocessed directory.
```
jupyter notebook preprocess.ipynb
```
2. Train the neural network with scripts/learning.py. Trained model and weights will be created in outputs directory.
```
python scripts/learning.py
```
3. If you like to predict rates for unknown MT stations, then run scripts/predict.py. The predicted rate will be in preprocessed/y_unrated.csv.
```
python scripts/predict.py
```

## License

## Problems/Questions
[Report an issue using the GitHub issue tracker](https://github.com/nimamura/pyMAGIQ/issues).

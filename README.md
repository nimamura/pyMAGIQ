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

## Run code
1. Please run pre-process by using notebooks/Preprocess.ipynb. Once pre-process has done, you will get npy-files in preprocessed directory.
```
jupyter notebook preprocess.ipynb
```
2. Train the neural network with scripts/Learning.py. Trained model and weights will be created in outputs directory.
```
python scripts/Learning.py
```
3. If you like to predict rates for unknown MT stations, then run scripts/predict.py. The predicted rate will be in preprocessed/y_unrated.csv.
```
python scripts/predict.py
```

## License

## Problems/Questions
[Report an issue using the GitHub issue tracker](https://github.com/nimamura/pyMAGIQ/issues).

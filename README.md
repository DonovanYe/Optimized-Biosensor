# Optimized-Biosensor

Codebase for Caltech CS101 Fall Term Rockley Biosensor project

Caltech-specific Drive can be found here (https://drive.google.com/drive/folders/11NOC_AVTIbZVXqcplVU5nI-KjhPXodyc?usp=share_link)

# Methods

## Codesign and PyTorch training in general
Consists of a variety of modules found in RockleyCodesign/codesign folder

### Data
Data is loaded via the data/rockley_data.py module.

### Sampler
A way to sample/undersample data.
 - IdentitySampler does nothing to data
 - LOUPESampler employs codesign sampling (see code for more details)
 - PreindexedSampler samples only the provided indices, setting the rest to 0
 - Subsampler samples only the provided indices, omitting the rest altogether
 - SetLOUPESampler is a version of LOUPESampler that tries to ensure exactly {budget} many lasers are selected where {budget} is a hyperparameter
 
### Predictor
 - NonlinearRegressor is a dense NN
 - CNNRegressor is a CNN built off DeepSpectra (may need to adjust dense layer if you adjust input size from 50)

Codesign involves using LOUPESampler with NonlinearRegressor (or SetLOUPESampler with CNNRegressor, but results for these are mixed)

See demo.ipynb for more on how to run

## Classification
Code can be found in ClassificationLogisticRegression notebook in notebooks folder


Data files are too big. Temporary fix is to just manually download from [here](https://drive.google.com/drive/folders/1vMvwF9VvCTEXDjc1W-Kfu5-y3x9ywpQ6?usp=share_link) and upload to data folder (will be ignored on commits/pushes)


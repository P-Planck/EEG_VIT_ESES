## Test on both Win10- Version	10.0.19045 Build 19045/Ubuntu 24.04.1 

### Create enviorment 

````
conda create -n EGG python==3.11
````

### require packages

````
torch
torchvision
math
matplotlib
````

````
conda install --yes --file require.txt
````

### form the dataset as img 

````
python ./data_pre/jpg_f.py
````

### training_process 

````
python ./vision_transformer/train.py
````

### test the predict process
````
python ./vision_transformer/predict.py
````










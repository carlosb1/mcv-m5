# Scene Understanding for Autonomous Vehicles
The objective of this project is the evaluation of the most well-known methods in the object detection / recognition / classification using the most popular neural network techniques.  To do this, we analyse different neural networks and configuration to do the review some the best methods. 

## Authors
Coen Antens

Carlos Báez

## Documentation
TODO 
- Paper
- Slides

## Results
link: https://owncloud.cvc.uab.es/owncloud/index.php/s/yCnAtuabr3s5cPa

## Getting started
### Prerequistes
In order to follow the practicum guide, we installed needed tools and created a softlink to reference our dataset:
```
ln -s access_module /share/mastergpu
```
### Experiments 
After this, the main objective was the execution of the provided tools from the git repository to analyse results:

1.- First step before starting to run our script, it is understand how works:

```
$ train.py --help

usage: train.py [-h] [-c CONFIG_PATH] [-e EXP_NAME] [-s SHARED_PATH]
                [-l LOCAL_PATH]

Model training

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_PATH, --config_path CONFIG_PATH
                        Configuration file
  -e EXP_NAME, --exp_name EXP_NAME
                        Name of the experiment
  -s SHARED_PATH, --shared_path SHARED_PATH
                        Path to shared data folder
  -l LOCAL_PATH, --local_path LOCAL_PATH
                        Path to local data folder
```

If we analyse the script. It is easy to test each required step in the practicum. 

- -c It specifies the modified configuration file (All the configuration files are modifications from `tt100k_classif.py` )
- -s It specifies the root folder from extract the necessary datasets for the experiments.
- -l It specifies the path for our workspace. It will have datasets and results from the executed experiments.
- -e It specifies the experiment name.

For our set of tests we apply create a softlink to easy the command line:
```
~$ ln -s /share/master access_modules
```
and we defined the workspace / local directory like `results`

After understanding this, each experiment is only change some parameters:

##### Run the provided code

- Analyse for crop and resize configuration: 
```
python train.py -c ./config/tt100k_classif_crop.py -l results -e experiment_crop -s ~/access_modules
python train.py -c ./config/tt100k_classif_resize.py -l results -e experiment_resize -s ~/access_modules
```

- Substract and mean configuration:
```
python train.py -c ./config/tt100k_classif_mean_substract.py -l results -e experiment_mean_substract -s ~/access_modules
```
 
- Fine tuning process:
TODO. Explains more related base model which it was applied to fine tuning
```
python train.py -c ./config/tt100k_classif_belgium.py -l results -e experiment_fine_tuning -s ~/access_modules
```



##### Train a network on another dataset

TODO pending to finish!!

##### Implement a new network
For the practicum, It was implemented the InceptionV3 network. Keras has support for this neural network, It was easy to add this 
network to our code. To do this, It was modified a few  main python classes.
The framework implements a factory pattern to load the different available models.

`model_factory.py` includes the method which set up a network model. Before to add the necessary line codes to add our model, we will create it which the provided structure data from `model.py`. In the `inceptionV3.py` and `squeezenet.py` is added the model. After this, in the `model.py` is added in the `make` function for our new networks. 

In order to execute the new neural network, it is created a new configuration file and execute: 
```
python train.py -c ./config/tt100k_classif_inceptionV3.py -l results -e experiment_inception -s ~/access_modules
python train.py -c ./config/tt100k_classif_squeezenet.py -l results -e experiment_squeezenet -s ~/access_modules

```
## Goals
 - [x] Run the provided code
 - [x] Train a network on another dataset
 - [x] Implement a new network
 - [ ] Implement a new network
 - [ ] Try to boost the performance of your network
 - [x] Report
 
 
## References
squeezenet implementation: https://github.com/rcmalli/keras-squeezenet


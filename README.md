# Scene Understanding for Autonomous Vehicles

## Authors

The objective of this project is the evaluation of the most well-known methods in the object detection / recognition / classification using the most popular neural network techniques.  To do this, we analyse different neural networks and configuration to do the review some the best methods. 

## Documentation
TODO 
- Paper
- Slides

## Results
TODO
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

- Analyse for crop and resize configuration: 
```
python train.py -c ./config/tt100k_classif_crop.py -l results -e experiment_crop -s ~/access_modules
python train.py -c ./config/tt100k_classif_resize.py -l results -e experiment_resize -s ~/access_modules
```
##### Description


- Analyse for crop and resize configuration: 
```
python train.py -c ./config/tt100k_classif_crop.py -l results -e experiment_crop -s ~/access_modules
python train.py -c ./config/tt100k_classif_resize.py -l results -e experiment_resize -s ~/access_modules
```
##### Description










### WARNING:

## Goals
 - [x]  yes
 - [ ]  no
 TODO

## Resources
TODO
## References
TODO


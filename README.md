# L2C
This repo contains codes for reproducing the experiments in the paper Learning to Counter: Stochastic Feature-based Learning for Diverse Counterfactual Explanations. 

## Dependencies 
`torch`, `sklearn` and `pandas==1.3.5` 

## Data
The folder `data/` contains train/test splits for each dataset : `German Credit (german)`, `Graduate Admission (admission)`, `Student Performance (student)` and `Small Business Administration (sba)`. 

Our experiment setup is described in `data_generator.py` and `data_loader.py`. 

## Black-box Models 
Each dataset has its own directory in `data/` and `model/` 
```
for name in german admission student sba
do
  mkdir model/$name/
  mkdir data/$name/
done
```
Our pre-trained models can be publicly accessed [here](https://drive.google.com/drive/folders/16wIKVHpf6n3CAWYlLaETJQ_uFHlaNI8H?usp=sharing). Each zip file contains 5 pre-trained models for each dataset. Download and place them in the corresponding folders. 

## Running L2C
To train the generative model, run the following command

```
python main.py train <dataset-name>
```

Please specify the correct dataset name given in the parentheses above. If there is no pre-trained black-box model, it will be trained before L2C model is run. Refer to `data_loader.py` and `blackbox.py` for details.

## Baselines 
Reproduction for `FastAR` can be found in `FastAR/`. 
Refer to [FastAR repo](https://github.com/vsahil/FastAR-RL-for-generating-AR) for more explanations.

First, you need to transfer our pre-processed data and pre-trained black-box in `FastAR/fastar/datasets` and `FastAR/fastar/models`  by running 

```
python compare.py fastar None
``` 

Next, go the this directory. Install dependencies and set up the sub-directories
```
cd FastAR
bash install_requirements.sh
cd fastar
for name in german admission student sba
do
  mkdir output/samples/$name/
  mkdir output/trained_models/$name/
done
```

You can find the base scripts `run_<dataset-name>.sh` for training and evaluation. Since `FastAR` only supports single class, 
the variable `$y` is to specify the desired class. Before running the script, you need to adjust line 61 and line 76 in `gym-midline/gym_midline/envs/my_<dataset-name>.py` accordingly. The current setup assumes label 0 is undesirable class and 1 is the desired outcome. The outputs will be written in `FastAR/fastar/output/`.

```
bash run_german.sh train
bash run_german.sh eval
```
In `baselines/`, we provide our customized codes for other baselines, gratefully borrowed from [DiCE repo](https://github.com/interpretml/DiCE), [MCCE repo](https://github.com/NorskRegnesentral/mccepy), [COPA repo](https://github.com/ngocbh/COPA) and [Certifai repo](https://github.com/Ighina/CERTIFAI). `Feasible-VAE` is supported in `DiCE` library. To run and evaluate the DiCE, F-VAE, COPA, MCCE and Certifai, 

```
python compare.py <method-name> <dataset-name>
```

`CRUDS` can be implemented off-the-shelf from [CARLA repo](https://github.com/carla-recourse/CARLA/tree/d9dd5740b54384e869b3fd48c82f52fb4ab39a93). 



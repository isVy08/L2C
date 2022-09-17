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
We provide our customized codes for some baselines, gratefully borrowed from [DiCE repo](https://github.com/interpretml/DiCE), [MCCE repo](https://github.com/NorskRegnesentral/mccepy), [COPA repo](https://github.com/ngocbh/COPA) and [Certifai repo](https://github.com/Ighina/CERTIFAI). `Feasible-VAE` is supported in `DiCE` library. To run and evaluate the DiCE, F-VAE, COPA, MCCE and Certifai, 

```
python compare.py <method-name> <dataset-name>
```

`CRUDS` can be implemented off-the-shelf from [CARLA repo](https://github.com/carla-recourse/CARLA/tree/d9dd5740b54384e869b3fd48c82f52fb4ab39a93). 

`FastAR` is a bit more complicated. First, clone the [FastAR repo](https://github.com/vsahil/FastAR-RL-for-generating-AR).
Transfer our pre-processed data and pre-trained black-box in `FastAR/fastar/datasets` and `FastAR/fastar/models`  by running 

```
python compare.py fastar None
``` 
The environment setup for our tasks are given in `baselines/gym-midline` and `baselines/classifer_dataset`. 
Replace FastAR original setups with (or include them in) the file/folder with the same names. 

To train FastAR on the first black-box model e.g., on German Credit, run the command

```
python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/german/mygerman1 --env-name gym_midline:mygerman-v1 --num-env-steps 100000 --lr 0.001
```

or evaluate by running
```
python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/german/mygerman1 --env-name gym_midline:mygerman-v1 --num-env-steps 100000 --lr 0.001 --eval
```

The outputs can be found in `FastAR/fastar/output/`.

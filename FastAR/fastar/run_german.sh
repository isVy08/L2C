#! /bin/bash
action=$1
y=0 #desired, opposite with below
# loop over sample
for i in 1..100
do
  if [ $action == "train" ]
  then 
    echo "TRAINING BEGINS FOR SAMPLE $i"
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/german/mygerman1$i --env-name gym_midline:mygerman-v1 --num-env-steps 100000 --lr 0.0001 --label $y
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/german/mygerman2$i --env-name gym_midline:mygerman-v2 --num-env-steps 100000 --lr 0.0001 --label $y
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/german/mygerman3$i --env-name gym_midline:mygerman-v3 --num-env-steps 100000 --lr 0.0001 --label $y
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/german/mygerman4$i --env-name gym_midline:mygerman-v4 --num-env-steps 100000 --lr 0.0001 --label $y
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/german/mygerman5$i --env-name gym_midline:mygerman-v5 --num-env-steps 100000 --lr 0.0001 --label $y
    
  else
    echo "EVALUATION BEGINS FOR SAMPLE $i"
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/german/mygerman1$i --env-name gym_midline:mygerman-v1 --eval 1 --output-dir ./output/samples/german/cfs1$i.csv --label $y
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/german/mygerman2$i --env-name gym_midline:mygerman-v2 --eval 1 --output-dir ./output/samples/german/cfs2$i.csv --label $y
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/german/mygerman3$i --env-name gym_midline:mygerman-v3 --eval 1 --output-dir ./output/samples/german/cfs3$i.csv --label $y
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/german/mygerman4$i --env-name gym_midline:mygerman-v4 --eval 1 --output-dir ./output/samples/german/cfs4$i.csv --label $y
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/german/mygerman5$i --env-name gym_midline:mygerman-v5 --eval 1 --output-dir ./output/samples/german/cfs5$i.csv --label $y
    
  fi  
done
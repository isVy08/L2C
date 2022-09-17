#! /bin/bash
action=$1
y=0 # desired label
# loop over sample of undesired
for i in 1..100
do
  if [ $action == "train" ]
  then 
    echo "TRAINING BEGINS FOR SAMPLE $i FOR"
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/admission/myadmission1$i --env-name gym_midline:myadmission-v1 --num-env-steps 100000 --lr 0.001 --label $y
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/admission/myadmission2$i --env-name gym_midline:myadmission-v2 --num-env-steps 100000 --lr 0.001 --label $y
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/admission/myadmission3$i --env-name gym_midline:myadmission-v3 --num-env-steps 100000 --lr 0.001 --label $y
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/admission/myadmission4$i --env-name gym_midline:myadmission-v4 --num-env-steps 100000 --lr 0.001 --label $y
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/admission/myadmission5$i --env-name gym_midline:myadmission-v5 --num-env-steps 100000 --lr 0.001 --label $y
    
  else
    echo "EVALUATION BEGINS FOR SAMPLE $i"
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/admission/myadmission1$i --env-name gym_midline:myadmission-v1 --eval 1 --output-dir ./output/samples/admission/cfs1$i.csv --label $y
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/admission/myadmission2$i --env-name gym_midline:myadmission-v2 --eval 1 --output-dir ./output/samples/admission/cfs2$i.csv --label $y
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/admission/myadmission3$i --env-name gym_midline:myadmission-v3 --eval 1 --output-dir ./output/samples/admission/cfs3$i.csv --label $y
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/admission/myadmission4$i --env-name gym_midline:myadmission-v4 --eval 1 --output-dir ./output/samples/admission/cfs4$i.csv --label $y
    python -W ignore main.py --entropy-coef 0.01 --save-dir ./output/trained_models/admission/myadmission5$i --env-name gym_midline:myadmission-v5 --eval 1 --output-dir ./output/samples/admission/cfs5$i.csv --label $y
    
  fi  
done

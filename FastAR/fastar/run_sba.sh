action=$1
y=0 #desired, opposite with below
# loop over sample
for i in 1..100
do
  if [ $action == "train" ]
  then 
    echo "TRAINING BEGINS FOR SAMPLE $i"
    python -W ignore main.py --entropy-coef 1.0 --save-dir ./output/trained_models/sba/mysba1$i --env-name gym_midline:mysba-v1 --num-env-steps 100000 --lr 0.0001 --label $y
    python -W ignore main.py --entropy-coef 1.0 --save-dir ./output/trained_models/sba/mysba2$i --env-name gym_midline:mysba-v2 --num-env-steps 100000 --lr 0.0001 --label $y
    python -W ignore main.py --entropy-coef 1.0 --save-dir ./output/trained_models/sba/mysba3$i --env-name gym_midline:mysba-v3 --num-env-steps 100000 --lr 0.0001 --label $y
    python -W ignore main.py --entropy-coef 1.0 --save-dir ./output/trained_models/sba/mysba4$i --env-name gym_midline:mysba-v4 --num-env-steps 100000 --lr 0.0001 --label $y
    python -W ignore main.py --entropy-coef 1.0 --save-dir ./output/trained_models/sba/mysba5$i --env-name gym_midline:mysba-v5 --num-env-steps 100000 --lr 0.0001 --label $y
    
  else
    echo "EVALUATION BEGINS FOR SAMPLE $i"
    python -W ignore main.py --entropy-coef 1.0 --save-dir ./output/trained_models/sba/mysba1$i --env-name gym_midline:mysba-v1 --eval 1 --output-dir ./output/samples/sba/cfs1$i.csv --label $y
    python -W ignore main.py --entropy-coef 1.0 --save-dir ./output/trained_models/sba/mysba2$i --env-name gym_midline:mysba-v2 --eval 1 --output-dir ./output/samples/sba/cfs2$i.csv --label $y
    python -W ignore main.py --entropy-coef 1.0 --save-dir ./output/trained_models/sba/mysba3$i --env-name gym_midline:mysba-v3 --eval 1 --output-dir ./output/samples/sba/cfs3$i.csv --label $y
    python -W ignore main.py --entropy-coef 1.0 --save-dir ./output/trained_models/sba/mysba4$i --env-name gym_midline:mysba-v4 --eval 1 --output-dir ./output/samples/sba/cfs4$i.csv --label $y
    python -W ignore main.py --entropy-coef 1.0 --save-dir ./output/trained_models/sba/mysba5$i --env-name gym_midline:mysba-v5 --eval 1 --output-dir ./output/samples/sba/cfs5$i.csv --label $y
    
  fi  
done
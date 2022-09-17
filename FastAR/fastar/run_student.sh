action=$1
y=0 #desired, opposite with below
# loop over sample
for i in 1..100
do
  if [ $action == "train" ]
  then 
    echo "TRAINING BEGINS FOR SAMPLE $i"
    python -W ignore main.py --entropy-coef 0.1 --save-dir ./output/trained_models/student/mystudent1$i --env-name gym_midline:mystudent-v1 --num-env-steps 100000 --lr 0.0001 --label $y
    python -W ignore main.py --entropy-coef 0.1 --save-dir ./output/trained_models/student/mystudent2$i --env-name gym_midline:mystudent-v2 --num-env-steps 100000 --lr 0.0001 --label $y
    python -W ignore main.py --entropy-coef 0.1 --save-dir ./output/trained_models/student/mystudent3$i --env-name gym_midline:mystudent-v3 --num-env-steps 100000 --lr 0.0001 --label $y
    python -W ignore main.py --entropy-coef 0.1 --save-dir ./output/trained_models/student/mystudent4$i --env-name gym_midline:mystudent-v4 --num-env-steps 100000 --lr 0.0001 --label $y
    python -W ignore main.py --entropy-coef 0.1 --save-dir ./output/trained_models/student/mystudent5$i --env-name gym_midline:mystudent-v5 --num-env-steps 100000 --lr 0.0001 --label $y
    
  else
    echo "EVALUATION BEGINS FOR SAMPLE $i"
    python -W ignore main.py --entropy-coef 0.1 --save-dir ./output/trained_models/student/mystudent1$i --env-name gym_midline:mystudent-v1 --eval 1 --output-dir ./output/samples/student/cfs1$i.csv --label $y
    python -W ignore main.py --entropy-coef 0.1 --save-dir ./output/trained_models/student/mystudent2$i --env-name gym_midline:mystudent-v2 --eval 1 --output-dir ./output/samples/student/cfs2$i.csv --label $y
    python -W ignore main.py --entropy-coef 0.1 --save-dir ./output/trained_models/student/mystudent3$i --env-name gym_midline:mystudent-v3 --eval 1 --output-dir ./output/samples/student/cfs3$i.csv --label $y
    python -W ignore main.py --entropy-coef 0.1 --save-dir ./output/trained_models/student/mystudent4$i --env-name gym_midline:mystudent-v4 --eval 1 --output-dir ./output/samples/student/cfs4$i.csv --label $y
    python -W ignore main.py --entropy-coef 0.1 --save-dir ./output/trained_models/student/mystudent5$i --env-name gym_midline:mystudent-v5 --eval 1 --output-dir ./output/samples/student/cfs5$i.csv --label $y
    
  fi  
done
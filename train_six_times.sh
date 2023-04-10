for n in {1..6};
do
    python active_learning_exp.py --strategy $1 --exp_name $2 --use_wandb $3
done
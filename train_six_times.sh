for n in {1..6};
do
    python main_active_learning.py --strategy $1 --exp_name $2 --use_wandb $3
done
for n in {1..6};
do
    python active_learning_coreset.py --strategy coreset_en --n_epochs 100 --batch_size 16 --exp_name 'Coreset-en'
done
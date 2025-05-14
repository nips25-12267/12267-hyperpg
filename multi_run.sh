for dataset in birds
do
    for seed in 123 #456 789
    do
        python main.py --config configs/baseline/baseline_resnet50_$dataset.py --config.seed=$seed
        python main.py --config configs/ppn_resnet50_$dataset.py --config.model.pm_name=l2 --config.training.num_epochs=200 --config.seed=$seed
        
        python main.py --config configs/hyperpg_resnet50_$dataset.py --config.model.pm_name=hyperpg --config.training.num_epochs=30 --config.seed=$seed
        python main.py --config configs/hyperpg_resnet50_$dataset.py --config.model.pm_name=cosine --config.training.num_epochs=30 --config.seed=$seed
        python main.py --config configs/hyperpg_resnet50_$dataset.py --config.model.pm_name=gaussian --config.training.num_epochs=100 --config.seed=$seed
    done
done

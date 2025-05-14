python main.py --config=configs/ppn_resnet50_birds.py --config.model.pm_name=vmf --config.training.num_epochs=50
python main.py --config=configs/ppn_resnet50_birds.py --config.model.pm_name=vmf --config.training.num_epochs=50 --config.model.latent_channels=64
python main.py --config=configs/ppn_resnet50_birds.py --config.model.pm_name=vmf --config.training.num_epochs=50 --config.model.latent_channels=32

python main.py --config=configs/ppn_resnet50_cars.py --config.model.pm_name=vmf --config.training.num_epochs=50
python main.py --config=configs/ppn_resnet50_cars.py --config.model.pm_name=vmf --config.training.num_epochs=50 --config.model.latent_channels=64
python main.py --config=configs/ppn_resnet50_cars.py --config.model.pm_name=vmf --config.training.num_epochs=50 --config.model.latent_channels=32

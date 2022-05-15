# al_base
python tasks/run_active_learning_training.py --backbone_name vinai/bertweet-covid19-base-cased --method random --max_epochs 100 --lr 1e-5 --batch_size 32 --gpus 0, &
python tasks/run_active_learning_training.py --backbone_name vinai/bertweet-covid19-base-cased --method entropy --max_epochs 100 --lr 1e-5 --batch_size 32 --gpus 1, &
python tasks/run_active_learning_training.py --backbone_name vinai/bertweet-covid19-base-cased --method margin --max_epochs 100 --lr 1e-5 --batch_size 32 --gpus 2, &
python tasks/run_active_learning_training.py --backbone_name vinai/bertweet-covid19-base-cased --method lc --max_epochs 100 --lr 1e-5 --batch_size 32 --gpus 3, &

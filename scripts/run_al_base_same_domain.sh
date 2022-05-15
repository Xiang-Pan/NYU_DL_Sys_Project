# al_base
python tasks/run_active_learning_training.py --backbone_name deepset/roberta-base-squad2 --method random --max_epochs 100 --lr 1e-5 --batch_size 32 --gpus 0, &
python tasks/run_active_learning_training.py --backbone_name deepset/roberta-base-squad2 --method entropy --max_epochs 100 --lr 1e-5 --batch_size 32 --gpus 1, &
python tasks/run_active_learning_training.py --backbone_name deepset/roberta-base-squad2 --method margin --max_epochs 100 --lr 1e-5 --batch_size 32 --gpus 2, &
python tasks/run_active_learning_training.py --backbone_name deepset/roberta-base-squad2 --method lc --max_epochs 100 --lr 1e-5 --batch_size 32 --gpus 3, &

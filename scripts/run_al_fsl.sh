# base
python tasks/run_active_learning_training_fsl.py --backbone_name roberta-base --method random --max_epochs 100 --lr 1e-6 --batch_size 16 --gpus 5, &
python tasks/run_active_learning_training_fsl.py --backbone_name roberta-base --method entropy --max_epochs 100 --lr 1e-6 --batch_size 16 --gpus 5, &
python tasks/run_active_learning_training_fsl.py --backbone_name roberta-base --method margin --max_epochs 100 --lr 1e-6 --batch_size 16 --gpus 5,  &
python tasks/run_active_learning_training_fsl.py --backbone_name roberta-base --method lc --max_epochs 100 --lr 1e-6 --batch_size 16 --gpus 5,      & 

wait;
# domain 
python tasks/run_active_learning_training_fsl.py --backbone_name vinai/bertweet-covid19-base-cased --method random --max_epochs 100 --lr 1e-6 --batch_size 16 --gpus 5, &
python tasks/run_active_learning_training_fsl.py --backbone_name vinai/bertweet-covid19-base-cased --method entropy --max_epochs 100 --lr 1e-6 --batch_size 16 --gpus 5, &
python tasks/run_active_learning_training_fsl.py --backbone_name vinai/bertweet-covid19-base-cased --method margin --max_epochs 100 --lr 1e-6 --batch_size 16 --gpus 5,  &
python tasks/run_active_learning_training_fsl.py --backbone_name vinai/bertweet-covid19-base-cased --method lc --max_epochs 100 --lr 1e-6 --batch_size 16 --gpus 5,      & 

wait;
# task
python tasks/run_active_learning_training_fsl.py --backbone_name deepset/roberta-base-squad2 --method random --max_epochs 100 --lr 1e-6 --batch_size 16 --gpus 5, &
python tasks/run_active_learning_training_fsl.py --backbone_name deepset/roberta-base-squad2 --method entropy --max_epochs 100 --lr 1e-6 --batch_size 16 --gpus 5, &
python tasks/run_active_learning_training_fsl.py --backbone_name deepset/roberta-base-squad2 --method margin --max_epochs 100 --lr 1e-6 --batch_size 16 --gpus 5,  &
python tasks/run_active_learning_training_fsl.py --backbone_name deepset/roberta-base-squad2 --method lc --max_epochs 100 --lr 1e-6 --batch_size 16 --gpus 5,      & 

wait;
# all
python tasks/run_active_learning_training_fsl.py --backbone_name deepset/roberta-base-squad2-covid --method random --max_epochs 100 --lr 1e-6 --batch_size 16 --gpus 5, &
python tasks/run_active_learning_training_fsl.py --backbone_name deepset/roberta-base-squad2-covid --method entropy --max_epochs 100 --lr 1e-6 --batch_size 16 --gpus 5, &
python tasks/run_active_learning_training_fsl.py --backbone_name deepset/roberta-base-squad2-covid --method margin --max_epochs 100 --lr 1e-6 --batch_size 16 --gpus 5,  &
python tasks/run_active_learning_training_fsl.py --backbone_name deepset/roberta-base-squad2-covid --method lc --max_epochs 100 --lr 1e-6 --batch_size 16 --gpus 5,      & 




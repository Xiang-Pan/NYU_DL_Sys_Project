# NYU_DL_Sys_Project

# Results
## Upper bound (Same Domain Fully Finetune)
```
python covid_cls_finetune.py --max_epochs 100 --batch_size 32 --lr 1e-5
```
Validation Results (F1): 0.8214

## Same Domain Transfer
squad2 Pertrained Model (Randomly Initialized CLS)  
Validation Results (F1): 0.0446
```
python tasks/test_transfer_model.py --backbone_name "deepset/roberta-base-squad2-covid"
```
```
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test/acc          │    0.0446428582072258     │
│          test/f1          │    0.0446428582072258     │
│         test/loss         │     2.783973217010498     │
└───────────────────────────┴───────────────────────────┘
```

squad2 Pertrained Model (CLS From fine-tuned model)  
```
python tasks/test_transfer_model.py --backbone_name "deepset/roberta-base-squad2-covid" --load_classifier "cached_models/covidqcls_finetune/fine_tune/epoch=20-val_f1=0.8393_classifier.pt"
```
```
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test/acc          │    0.0803571417927742     │
│          test/f1          │    0.0803571417927742     │
│         test/loss         │    2.8527073860168457     │
└───────────────────────────┴───────────────────────────┘
```
squad2 Pertrained Model (CLS From fine-tuned model)
```
python tasks/test_transfer_model.py --backbone_name "deepset/roberta-base-squad2-covid" --load_classifier "cached_models/covidqcls_finetune/fine_tune/epoch=20-val_f1=0.8393_classifier.pt"
```
```
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test/acc          │    0.0803571417927742     │
│          test/f1          │    0.0803571417927742     │
│         test/loss         │    2.8527073860168457     │
└───────────────────────────┴───────────────────────────┘
```

## FewShot
```
python tasks/covid_cls_few_shots.py --backbone_name "deepset/roberta-base-squad2-covid"  --num_shots 1 --log_name squad2-covid_1shot --batch_size 16 --lr 1e-6
```
### One-shot (1g9wecae)

Validation Results (F1): 0.2232, epoch = 55

### Two-shot (bh75dtne)
Validation Results (F1): 0.29, epoch = 55

### Three-shot ()
Validation Results (F1): 0.29, epoch = 55

### Four-shot ()
Validation Results (F1): 0.29, epoch = 55

### Five-shot ()
Validation Results (F1): 0.29, epoch = 55

## aaa


## Other
CovidNER Pertrained Model (CLS From fine-tuned model)  
```
```


CovidNER Pertrained Model (One-Shot: One Example)
```
```


CovidNER Pertrained Model (One-Shot: One Example)
```
```


## Same Task Transfer

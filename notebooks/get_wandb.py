import pandas as pd 
import wandb
import json
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("xiang-pan/NYU_DL_Sys_Project")


name2id = {}

summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    
    wandb_id = run.id.split("/")[-1]
    print(wandb_id)
    # .name is the human-readable name of the run.
    name_list.append(run.name)
    if "deepset/roberta-base-squad2-covid" in run.name:
        name = run.name.replace("deepset/roberta-base-squad2-covid", "task-domain")
    elif "vinai/bertweet-covid19-base-cased" in run.name:
        name = run.name.replace("vinai/bertweet-covid19-base-cased", "domain")
    elif "deepset/roberta-base-squad2" in run.name:
        name = run.name.replace("deepset/roberta-base-squad2", "task")
    else:
        name = run.name.replace("roberta-base", "baseline")
    name2id[name] = wandb_id

    

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

runs_df.to_csv("project.csv")


# save the name2id dict
with open("name2id.json", "w") as f:
    json.dump(name2id, f)
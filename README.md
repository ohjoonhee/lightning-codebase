# Intro
This is to-go pytorch template utilizing [lighting](https://github.com/Lightning-AI/lightning) and [wandb](https://github.com/wandb/wandb). 
This template uses `Lightning CLI` for config management. 
It follows most of [Lightning CLI docs](https://lightning.ai/docs/pytorch/latest/api_references.html#cli) but, integrated with `wandb`.
Since `Lightning CLI` instantiate classes on-the-go, there were some work-around while integrating `WandbLogger` to the template.
This might **not** be the best practice, but still it works and quite convinient.

# How To Use
It uses `Lightning CLI`, so most of its usage can be found at its [official docs](https://lightning.ai/docs/pytorch/latest/api_references.html#cli).  
There are some added arguments related to `wandb`.

* `--name` or `-n`: Name of the run, displayed in `wandb`
* `--version` or `-v`: Version of the run, displayed in `wandb` as tags

Basic cmdline usage is same with the `main` branch.  
```bash
python src/main.py fit -c configs/config.yaml -n my_exp_name -v my_trial_version
```
The followings are `rl` branch specific usages.  

## Log Gym Env
Simply add the following arguments to `config.yaml` (only if `wandb` is used).  
```yaml
trainer:
  logger:
    init_args:
      monitor_gym: true
```

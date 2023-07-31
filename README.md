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

Basic cmdline usage is as follows.  
We assume cwd is project root dir.

### `fit` stage 
```bash
python src/main.py fit -c configs/config.yaml -n debug-fit-run -v debug-version
```
If using `wandb` for logging, change `"project"` key in `cli_module/rich_wandb.py`

#### Automatic Batch Size Finder
Just add `BatchSizeFinder` callbacks in the config
```yaml
trainer:
  callbacks:
    - class_path: BatchSizeFinder
```
Or add them in the cmdline.
```bash
python src/main.py fit -c configs/config.yaml --trainer.callbacks+=BatchSizeFinder
```

##### NEW! `tune.py` for lr_find and batch size find
```bash
python src/tune.py -c configs/config.yaml
```
NOTE: No subcommand in cmdline

#### Resume
Basically all logs are stored in `logs/${name}/${version}/${job_type}` where `${name}` and `${version}` are configured in yaml file or cmdline. 
`{job_type}` can be one of `fit`, `test`, `validate`, etc.

```
   
  

### `test` stage
```bash
python src/main.py test -c configs/config.yaml -n debug-test-run -v debug-version --ckpt_path YOUR_CKPT_PATH
```




## TODO
* Check pretrained weight loading
* Consider multiple optimizer using cases (i.e. GAN)
* Add instructions in README (on-going)
* Clean code
 
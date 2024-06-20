# Intro
This is to-go pytorch template utilizing [lightning](https://github.com/Lightning-AI/lightning) and [wandb](https://github.com/wandb/wandb). 
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
If you want to access log directory in your `LightningModule`, you can access as follows.
```python
log_root_dir = self.logger.log_dir or self.logger.save_dir
```

### Clean Up Wandb Artifacts
If using `wandb` for logging, model ckpt files are uploaded to `wandb`.  
Since the size of ckpt files are too large, clean-up process needed.  
Clean-up process delete all model ckpt artifacts without any aliases (e.g. `best`, `lastest`)
To toggle off the clean-up process, add the following to `config.yaml`. Then every version of model ckpt files will be saved to `wandb`.
```yaml
trainer:
  logger:
    init_args:
      clean: false
```

### Model Checkpoint
One can save model checkpoints using `Lightning Callbacks`. 
It contains model weight, and other state_dict for resuming train.  
There are several ways to save ckpt files at either local or cloud.

1. Just leave everything in default, ckpt files will be saved locally. (at `logs/${name}/${version}/fit/checkpoints`)

2. If you want to save ckpt files as `wandb` Artifacts, add the following config. (The ckpt files will be saved locally too.)
```yaml
trainer:
  logger:
    init_args:
      log_model: all
```
3. If you want to save ckpt files in cloud rather than local, you can change the save path by adding the config. (The ckpt files will **NOT** be saved locally.)
```yaml
model_ckpt:
  dirpath: gs://bucket_name/path/for/checkpoints
```

#### `AsyncCheckpointIO` Plugins
You can set async checkpoint saving by providing config as follows.  
```yaml
trainer:
  plugins:
    - AsyncCheckpointIO
```



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
  

### `test` stage
```bash
python src/main.py test -c configs/config.yaml -n debug-test-run -v debug-version --ckpt_path YOUR_CKPT_PATH
```




## TODO
* Check pretrained weight loading
* Consider multiple optimizer using cases (i.e. GAN)
* Add instructions in README (on-going)
* Clean code
 
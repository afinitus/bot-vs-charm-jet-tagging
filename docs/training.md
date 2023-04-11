Before getting started with this section, please make sure you have read the [preprocessing documentation](preprocessing.md) which describes how to produce training files for the framework.

### Setup Logging

Salt has the potential to supports any logging framework also supported by Lightning.
At the moment only comet is supported.

#### Comet

To use the [comet](https://www.comet.ml/) logger, you need to make an account with comet and [generate an API key](https://www.comet.ml/docs/quick-start/#getting-your-comet-api-key).
You also need to create a [workspace](https://www.comet.ml/docs/user-interface/#workspaces).
Next save the API key and the workspace name in environment variables called `COMET_API_KEY` and `COMET_WORKSPACE`.
These variables are automatically read by the comet, see [here](https://www.comet.com/docs/v2/guides/tracking-ml-training/configuring-comet/#configure-comet-through-environment-variables) for more info.
Consider adding these variables to your [bashrc](https://www.journaldev.com/41479/bashrc-file-in-linux).

??? info "Add the environment variable to your bashrc"

    To ensure the environment variables are defined every time you log in,
    you can add the definitions to your bashrc.
    Simply add the lines

    ```bash
    export COMET_API_KEY="<Your API Key>"
    export COMET_WORKSPACE="<Your Workspace Name>"
    ```

    to your `~/.bashrc` file.
    If no such file exists, create one in your home directory.

### Training

Training is fully configured via a YAML config file and a CLI powered by [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli.html#lightning-cli).
This allows you control all aspects of the training from config or directly via command line arguments.

The configuration is split into two parts.
The [`base.yaml`]({{repo_url}}-/blob/main/salt/configs/base.yaml) config contains model-independent information like the input file paths and batch size.
This config is used by default for all trainings without you having to explicitly specify it.
Meanwhile the model configs, for example [`gnn.yaml`]({{repo_url}}-/blob/main/salt/configs/gnn.yaml) contain a full description of a specific model, including a list of input variables used.
You can start a training for a given model by providing it as an argument to the `main.py` python script, which is also exposed through the command `salt`.

```bash
salt fit --config configs/GN1.yaml
```

The subcommand `fit` specifies you want to train the model, rather than [evaluate](evaluation.md) it.
It's possible to specify more than one configuration file, for example to override the values set in [`base.yaml`]({{repo_url}}-/blob/main/salt/configs/base.yaml).
The CLI will merge them [automatically](https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli_advanced.html#compose-yaml-files).

??? info "Running a test training"

    To test the training script you can run use the `--trainer.fast_dev_run` flag
    which will run over a small number of training and validation batches and then
    exit.

    ```bash
    salt fit --config configs/GN1.yaml --trainer.fast_dev_run 2
    ```

    Logging and checkpoint are suppressed when using this flag.


#### Using the CLI

You can also configure the training directly through CLI arguments.
For a full list of available arguments run

```bash
salt fit --help
```

#### Choosing GPUs

By default the config will try to use the first available GPU, but
you can specify which ones to use with the `--trainer.devices` flag.
Take a look [here](https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_basic.html#train-on-multiple-gpus) to learn more about the different ways you can specify which GPUs to use.

??? warning "Check GPU usage before starting training"

    You should check with `nvidia-smi` that any GPUs you use are not in use by some other user before starting training.

### Reproducibility

Reproducibility of tagger trainings is very important.
The following features try to ensure reproducibility without too much overhead.

#### Commit Hashes & Tags

When training, the framework will complain if you have any uncommitted changes.
You can override this by setting the `--force` flag, but you should only do so if you are sure that preserving your training configuration is not necessary.
The commit hash used for the training is saved in the `metadata.yaml` for the training run.

Additionally, you can create and push a tag of the current state using the `--tag` argument.
This will push a tag to your personal fork of the repo.
The tag name is the same as the output directory name, which is generated automatically by the framework.

??? warning "Using `--force` will prevent a tag from being created and pushed"

    Try and avoid using `--force` if you can, as it hampers reproducibility.


#### Random Seeds

Training runs are reproducible thanks to the `--seed_everything` flag,
which is already set for you in the [`base.yaml`]({{repo_url}}-/blob/main/salt/configs/base.yaml) config.
The flat seeds all random number generators used in the training.
This means for example that weight initialisation and data shuffling happen in a deterministic way.

??? info "Stochastic operations can still lead to divergences between training runs"

    For more info take a look [here](https://pytorch.org/docs/stable/notes/randomness.html).


### Dataloading

Jets are loaded in weakly shuffled batches from the training file.
This is much more efficient than randomly accessing individual jets, which would be prohibitively slow.

Some other dataloading considerations are discussed below.

#### Worker Counts

During training, data is loaded using worker processes.
The number of workers used is specified by `data.num_workers`.
Increasing the worker count will speed up training until you max out your GPUs processing capabilities.
Test different counts to find the optimal value, or just set this to the number of CPUs on your machine.

??? info "Maximum worker counts"

    You can find out the number of CPUs available on your machine by running

    ```bash
    cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1
    ```

    You should not use more workers than this.
    If you use too few or too many workers, you will see a warning at the start of training.

#### Fast Disk Access

Most HPC systems will have dedicated fast storage.
Loading training data from these drives can significantly improve training times.
To temporarily copy training files into a target directory before training, use the
`--data.move_files_temp=/temp/path/` flag.

If you have enough RAM, you can load the training data into shared memory before starting training by setting `move_files_temp` to a path under `/dev/shm/<username>`.

??? warning "Ensure temporary files are removed"

    The code will try to remove the temporary files when the training is complete, but if the training is interrupted this may not happen.
    You should double check whether you need to manually remove the temporary files to avoid clogging up your system's RAM.

### Slurm Batch

Those at institutions with Slurm managed GPU batch queues can submit training jobs using

```bash
sbatch submit_slurm.sh
```

The submit script only supports running from a conda environment for now.
There are several options in the script which need to be tailored to make sure to make a look inside.

??? info "Cleaning up after interruption"

    If training is interrupted, you can be left with floating worker processes on the node which can clog things up for other users.
    You should check for this after running training jobs which are cancelled or fail.
    To do so, `srun` into the affected node using

    ```bash
    srun --pty --cpus-per-task 2 -w compute-gpu-0-3 -p GPU bash
    ```

    and then kill any remaining running python processes using

    ```bash
    pkill -u <username> -f salt -e
    ```

### Resuming Training

Model checkpoints are saved in timestamped directories under `logs/`.
These directories also get a copy of the fully merging training config (`config.yaml`), and a copy of the umami scale dict.
To resume a training, point to a previously saved config and a `.ckpt` checkpoint file by using the `--ckpt_path` argument.

```bash
salt fit --config logs/run/config.yaml --ckpt_path path/to/checkpoint.ckpt
```

The full training state, including the state of the optimiser, is resumed.
The logs for the resumed training will be saved in a new directory, but the epoch count will continue from
where it left off.



### Customising Training

#### Switching Attention Mechanisms

By default the [`GN1.yaml`]({{repo_url}}-/blob/main/salt/configs/GN1.yaml) config uses the scaled dot product attention found in transformers.
To switch to the GATv2 attention, add the [`GATv2.yaml`]({{repo_url}}-/blob/main/salt/configs/GATv2.yaml) config fragment.

```bash
salt fit --config configs/GN1.yaml --config configs/GATv2.yaml
```

Note that the  `num_heads` and `head_dim` arguments must match those in the `gnn.init_args` config block.

#### Switching Pooling Mechanisms

The [`GN2.yaml`]({{repo_url}}-/blob/main/salt/configs/GN2.yaml) config by default uses Global Attention Pooling. This can be switched out for Cross Attention Pooling where a learnable class token is attended to by the learned track representation. The same arguments used for the Transformer Encoder apply to the Cross Attention layers. An example Cross Attention block is shown below:

```yaml
pool_net:
    class_path: salt.models.CrossAttentionPooling
    init_args:
        embed_dim: *embed_dim
        num_layers: 3
        out_dim: &out_dim 128
        mha_config:
            num_heads: 4
            attention:
              class_path: salt.models.ScaledDotProductAttention
            out_proj: False
        dense_config:
            norm_layer: *norm_layer
            activation: *activation
            hidden_layers: [128]
            dropout: 0.1

```

#### Selecting Training Variables

Training files are structured arrays, so it is easy to specify which variables you want to include in the training by name.

In your `data` config there is a `variables` key which specifies which variables to include in the training for each input type.
These are defined in the model files, rather than the `base.yaml` config.

??? warning "Make sure you do not train on truth information!"

    The variables listed under `data.variables` are the inputs to the training.
    You should include any truth information, but rather specify truth labels for each task in your model config.

For example, in [`GN1.yaml`]({{repo_url}}-/blob/main/salt/configs/GN1.yaml) you will find the following variables:

```yaml
data:
  variables:
    jet:
      - pt_btagJes
      - eta_btagJes
    track:
      - d0
      - z0SinTheta
      - dphi
      - deta
      ...
```

For now, you will need to manually match your model input size to the number of variables you are using.


#### Remapping Labels

In the config block for each task, you can specify a `label_map` which maps the labels in the training file to the labels you want to use in the training.
For example, 

```yaml
class_path: salt.models.ClassificationTask
init_args:
    input_type: jet
    name: jet_classification
    label: HadronConeExclTruthLabelID
    label_map: { 0: 0, 4: 1, 5: 2 }
    ...
```

#### Jet features concatentaion

You can concatenate the jet features with the node embeddings after the GNN step. In order to this you should add a `global` key to the `inputs` section and specify which variables do you want to concatenate in the `data.variables` section.

??? warning "Don't forget to change pooling and task input size accordingly"

    If you concatenate 2 variables you should increase the `input_size` by 2 (for example `128->130`) for `pool_net` and all tasks (except vertexing, here you should increase by 4).

For example, for the [`GN1.yaml`]({{repo_url}}-/blob/main/salt/configs/GN1.yaml) you can add jet features concatenation in the following way:

- `inputs` section
    ```yaml
    inputs:
        jet: jets
        track: tracks
        global: jets
    ```
- `variables` section
    ```yaml
    data:
    variables:
        global:
        - pt_btagJes
        - eta_btagJes
        jet:
        - pt_btagJes
        - eta_btagJes
        track:
        - d0
        - z0SinTheta
        - dphi
        - deta
        ...
    ```

You can find the full example at [`GN2emu.yaml`]({{repo_url}}-/blob/main/salt/configs/GN2Cat.yaml)
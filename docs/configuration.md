# Advanced Configuration


### Dataloading

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


#### Global Jet Features

By default, variables under the `jet` key are concatenated with each of the input tracks.
You can also chooose to instead concatenate jet features with the pooled jet embedding after the GNN step.
In order to this you should add a `global` key under `data.variables.inputs` and specify which variables do you want to use.

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


#### Edge Features

It is possible to include edge features as network input, representing relational information between constituent tracks. These have to be implemented on an individual basis since they are not stored in the input files, but rather calculated on the fly within Salt. As such, the information is drawn from the track container (or equivalent) and requires the presence of track variables relevant to the calculation of each edge feature. Currently implemented features are:

- `dR` = $log(\sqrt{d\eta^2 + d\phi^2})$ (requires `phi`, `eta`)
- `kt` = $log(min(p_T)\sqrt{d\eta^2 + d\phi^2})$ (requires `pt`)
- `z` = $log(\frac{min(p_T)}{\Sigma(p_T)})$ (requires `pt`, `phi`, `eta`)
- `isSelfLoop` = 1 if edge represents self-connection, 0 if not
- `subjetIndex` = 1 if tracks are part of same subjet, 0 if not (requires `subjetIndex`)


#### Truncating Inputs

You can truncate the number of tracks used for training.
This can be useful to speed up training.
To do so, pass `--data.num_inputs.tracks=10` to limit the training to the first ten tracks.
This can also be configured in yaml via

```yaml
data:
  num_inputs:
    track: 10
```

#### Remapping Labels

This section is about remapping labels on the fly, which is useful in case they are not already mapped to `0, 1, 2...`.
In the config block for each task, you can specify a `label_map` which maps the label values as stored in the input file to the ones you want to use in the task loss calculation.
For example, instead of using the pre-mapped `flavour_label`, you could directly train on `HadronConeExclTruthLabelID` using,

```yaml
class_path: salt.models.ClassificationTask
init_args:
    input_type: jet
    name: jet_classification
    label: HadronConeExclTruthLabelID
    label_map: { 0: 0, 4: 1, 5: 2 }
    ...
```



### Model Architecture

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

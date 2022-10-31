

### Dumping Training Samples

You can create training samples using the [training dataset dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/).
The default config file [`EMPFlow.json`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/-/blob/r22/configs/single-b-tag/EMPFlowGNN.json) has all the information required to train models with salt.
Note that predumped h5 samples are available [here](https://umami-docs.web.cern.ch/preprocessing/mc-samples/).


### Preprocessing with Umami

The h5 files are processed by the [umami framework](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tree/master/umami) to produce training files.
The umami framework handles jet selection, kinematic resampling, and input normalisation.
Use [this preprocessing config](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami-config-tags/-/blob/master/offline/PFlow-Preprocessing-GNN.yaml) and [this variable config](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configs/GNN_Variables.yaml) for the creation of train samples for salt.

For more information, take a look at the umami [docs](https://umami-docs.web.cern.ch/trainings/GNN-instructions/)

#### Preprocessing Requirements

1. Please ensure you run preprocessing with a recent version of umami that includes [!648](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/648) (i.e. versions >=0.15).

2. It is also recommend to set `concat_jet_tracks: True` in your preprocessing config. See the [umami docs](https://umami-docs.web.cern.ch/preprocessing/write_train_sample/#config-file) for more info.

3. Finally, it is recommended to produce the training samples with 16-bit floating point precision. To do this set `precision: float16` in your preprocessing config. Reducing the precision leads to significantly smaller filesizes and improved dataloading speeds while maintaining the same level of performance.

#### Creating the Validation Sample

Umami can write a validation sample for you.
See [here](https://umami-docs.web.cern.ch/preprocessing/write_train_sample/#writing-validation-samples).

#### Directory Structure

Please note, training files are suggested to follow a certain directory structure, which is based on the output structure of umami preprocessing jobs.

```bash
- base_dir/
    - train_sample_1/
        # umami configuration
        - PFlow-Preprocessing.yaml
        - PFlow-scale_dict.json
        - GNN_Variables.yaml

        # tdd output datasets
        - source/
            - tdd_output_ttbar/
            - tdd_output_zprime/

        # umami hybrid samples
        - prepared/
            - MC16d-inclusive_testing_ttbar_PFlow.h5
            - MC16d-inclusive_testing_zprime_PFlow.h5

        # umami preprocessed samples
        - preprocessed/
            - PFlow-hybrid-resampled_scaled_shuffled.h5
            - PFlow-hybrid-validation-resampled_scaled_shuffled.h5
```

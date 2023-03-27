from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.recfunctions import unstructured_to_structured as u2s
from pytorch_lightning import Callback, LightningModule, Trainer

from salt.utils.arrays import join_structured_arrays
from salt.utils.union_find import get_node_assignment


class PredictionWriter(Callback):
    def __init__(
        self,
        jet_variables: list = None,
        track_variables: list = None,
        write_tracks: bool = False,
        half_precision: bool = False,
        jet_classes: list = None,
    ) -> None:
        """Write test outputs to h5 file.

        Parameters
        ----------
        jet_variables : list
            List of jet variables to copy from test file
        track_variables : list
            List of track variables to copy from test file
        write_tracks : bool
            If true, write track outputs to file
        half_precision : bool
            If true, write outputs at half precision
        jet_classes : list
            List of flavour names in order of label values
        """
        super().__init__()

        self.jet_variables = jet_variables
        self.track_variables = track_variables
        self.write_tracks = write_tracks
        self.half_precision = half_precision
        self.jet_classes = jet_classes
        self.track_origin_cols = [
            "Pileup",
            "Fake",
            "Primary",
            "FromB",
            "FromBC",
            "FromC",
            "FromTau",
            "OtherSecondary",
        ]
        self.track_type_cols = [
            "NoTruth",
            "Other",
            "Pion",
            "Kaon",
            "Electron",
            "Muon",
        ]

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "test":
            return

        # inputs names
        self.jet = trainer.datamodule.inputs["jet"]
        self.track = trainer.datamodule.inputs["track"]

        # place to store intermediate outputs
        self.task_names = [task.name for task in pl_module.model.tasks]
        self.outputs: dict = {task: [] for task in self.task_names}
        self.mask: list = []

        # get test dataset
        self.ds = trainer.datamodule.test_dataloader().dataset
        self.file = self.ds.file
        self.num_jets = len(self.ds)

        # get jet class prediction column names
        train_file = trainer.datamodule.train_dataloader().dataset.file
        if not self.jet_classes:
            self.jet_classes = train_file[f"{self.jet}"].attrs["flavour_label"]
            jet_task = [t for t in pl_module.model.tasks if t.name == "jet_classification"][0]
            if jet_task.label_map is not None:  # TODO: extend to xbb
                d = {0: "ljets", 4: "cjets", 5: "bjets"}
                self.jet_classes = [d[x] for x in jet_task.label_map]
        assert self.jet_classes is not None
        self.jet_cols = [f"{pl_module.name}_p{c.split('jets')[0]}" for c in self.jet_classes]

        # get output path
        out_dir = Path(trainer._ckpt_path).parent
        out_basename = str(Path(trainer._ckpt_path).stem)
        stem = str(Path(self.ds.filename).stem)
        sample = split[3] if len(split := stem.split("_")) == 4 else stem
        fname = f"{out_basename}__test_{sample}.h5"
        self.out_path = Path(out_dir / fname)

        # decide whether to write tracks
        self.task_list = [task.name for task in pl_module.model.tasks]
        self.write_tracks = self.write_tracks and any(
            t in self.task_list for t in ["track_origin", "track_vertexing"]
        )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dl_idx):
        # append test outputs
        [self.outputs[task].append(outputs[0][task].cpu()) for task in self.task_names]
        if self.write_tracks:
            self.mask.append(outputs[1]["track"].cpu())  # TODO: don't hardcode "track"

    def on_test_end(self, trainer, pl_module):
        # concat test batches
        outputs = {task: torch.cat(self.outputs[task]) for task in self.task_names}

        # softmax jet classification outputs
        jet_class_preds = torch.softmax(outputs["jet_classification"], dim=-1)

        # create output jet dataframe
        precision_str = "f2" if self.half_precision else "f4"
        dtype = np.dtype([(n, precision_str) for n in self.jet_cols])
        jets = u2s(jet_class_preds.float().cpu().numpy(), dtype)

        if self.jet_variables is None:
            self.jet_variables = self.file[self.jet].dtype.names

        jets2 = self.file[self.jet].fields(self.jet_variables)[: self.num_jets]
        jets = join_structured_arrays((jets, jets2))

        if self.write_tracks:
            if self.track_variables is None:
                self.track_variables = self.file[self.track].dtype.names

            t = self.file[self.track].fields(self.track_variables)[: self.num_jets]
            mask = torch.cat(self.mask).unsqueeze(dim=-1)

            # add output for track classification
            if "track_origin" in self.task_list:
                t2 = outputs["track_origin"]
                t2 = F.softmax(mask_fill_flattened(t2, mask), dim=-1).numpy()
                t2 = t2.view(dtype=np.dtype([(name, "f4") for name in self.track_origin_cols]))
                t2 = t2.reshape(t2.shape[0], t2.shape[1])
                t = join_structured_arrays((t, t2))

            if "track_type" in self.task_list:
                t2 = outputs["track_type"]
                t2 = F.softmax(mask_fill_flattened(t2, mask), dim=-1).numpy()
                t2 = t2.view(dtype=np.dtype([(name, "f4") for name in self.track_type_cols]))
                t2 = t2.reshape(t2.shape[0], t2.shape[1])
                t = join_structured_arrays((t, t2))

            # add output for vertexing
            if "track_vertexing" in self.task_list:
                t2 = outputs["track_vertexing"]
                t2 = get_node_assignment(
                    t2, (~mask).squeeze(dim=-1).sum(dim=-1).tolist()
                )  # could switch this to running on individual batches if memory becomes an issue
                t2 = mask_fill_flattened(t2, mask).numpy()

                # convert to structured array
                t2 = t2.view(dtype=np.dtype([("VertexIndex", "f4")]))
                t2 = t2.reshape(t2.shape[0], t2.shape[1])
                t = join_structured_arrays((t, t2))

        # write to h5 file
        print("\n" + "-" * 100)
        if self.out_path.exists():
            print("Warning! Overwriting existing file.")

        with h5py.File(self.out_path, "w") as f:
            self.create_dataset(f, jets, self.jet, self.half_precision)
            if self.write_tracks:
                self.create_dataset(f, t, self.track, self.half_precision)

        print("Created output file", self.out_path)
        print("-" * 100, "\n")

    def create_dataset(self, f, a, name, half_precision):
        # convert down to float16
        if half_precision:

            def half(t):
                t = np.dtype(t)
                if t.kind == "f" and t.itemsize == 2:
                    return "f2"
                return t

            a = np.array(a, dtype=[(n, half(t)) for n, t in a.dtype.descr])

        # write
        f.create_dataset(name, data=a, compression="lzf")


# convert flattened array to shape of mask (ntracks, ...) -> (njets, maxtracks, ...)
def mask_fill_flattened(flat_array, mask_array):
    filled = torch.full(
        (mask_array.shape[0], mask_array.shape[1], flat_array.shape[1]), float("-inf")
    )
    start_index = end_index = 0

    for i in range(mask_array.shape[0]):
        end_index += (~mask_array[i]).sum().item()
        filled[i, : end_index - start_index] = flat_array[start_index:end_index]
        start_index = end_index

    return filled

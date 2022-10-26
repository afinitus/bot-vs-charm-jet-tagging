import glob
import re
from pathlib import Path

import numpy as np
from pytorch_lightning.cli import LightningCLI


def get_best_epoch(config_path: Path) -> Path:
    ckpt_dir = Path(config_path.parent / "ckpts")
    print("No --ckpt_path specified, looking for best checkpoint in", ckpt_dir)
    ckpts = glob.glob(f"{ckpt_dir}/*.ckpt")
    exp = r"(?<=val_loss=)(?:(?:\d+(?:\.\d*)?|\.\d+))"
    losses = [float(re.findall(exp, Path(ckpt).name)[0]) for ckpt in ckpts]
    ckpt = ckpts[np.argmin(losses)]
    print("Using checkpoint", ckpt)
    return ckpt


class SaltCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        if self.subcommand == "test":
            print("\n" + "-" * 100)

            # modify callbacks when testing
            self.save_config_callback = None
            self.config[f"{self.subcommand}.trainer.logger"] = False
            for c in self.config[f"{self.subcommand}.trainer.callbacks"]:
                if hasattr(c, "init_args") and hasattr(c.init_args, "refresh_rate"):
                    c.init_args.refresh_rate = 1

            # use the best epoch for testing
            if self.config[f"{self.subcommand}.ckpt_path"] is None:
                config = self.config[f"{self.subcommand}.config"]
                assert len(config) == 1
                best_epoch_path = get_best_epoch(Path(config[0].rel_path))
                self.config[f"{self.subcommand}.ckpt_path"] = best_epoch_path

            # ensure only one devices is used for testing
            n_devices = self.config[f"{self.subcommand}.trainer.devices"]
            if isinstance(n_devices, str) and int(n_devices) > 1:
                print("Setting --trainer.devices=1")
                self.config[f"{self.subcommand}.trainer.devices"] = "1"
            if isinstance(n_devices, list) and len(n_devices) > 1:
                raise ValueError("Testing requires --trainer.devices=1")

            print("-" * 100 + "\n")

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Dict , List, Optional

from monai.transforms import Invertd, SaveImaged,Transform,MapTransform

import monailabel
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.sam2.utils import is_sam2_module_available
from monailabel.tasks.activelearning.first import First
from monailabel.tasks.activelearning.random import Random
from monailabel.tasks.infer.bundle import BundleInferTask
from monailabel.tasks.scoring.epistemic_v2 import EpistemicScoring
from monailabel.tasks.train.bundle import BundleTrainTask
from monailabel.utils.others.generic import get_bundle_models, strtobool

logger = logging.getLogger(__name__)
class SelectLabelsd(MapTransform):
    """
    Custom transform to filter specific label indices from segmentation output.
    Operates on dictionary data format (standard for MONAI Label).
    """
    def __init__(self, keys: str, label_indices: List[int]):
        super().__init__(keys)
        self.label_indices = label_indices

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                # Assuming channel-first format [C, H, W, D]
                d[key] = d[key][self.label_indices, ...]
                
                # Update corresponding metadata if exists
                meta_key = f"{key}_meta_dict"
                if meta_key in d:
                    orig_labels = d[meta_key].get("labels")
                    if orig_labels:
                        d[meta_key]["labels"] = [orig_labels[i] for i in self.label_indices]
        return d

class CustomBundleInferTask(BundleInferTask):
    def __init__(self, bundle_path, conf, label_indices: Optional[List[int]] = None, **kwargs):
        super().__init__(bundle_path, conf, **kwargs)
        self.label_indices = label_indices

    def post_transforms(self, data=None):
        # Get original post transforms
        transforms = super().post_transforms(data)
        
        # Find position of AsDiscreted transform
        discreted_idx = -1
        for i, t in enumerate(transforms):
            if t.__class__.__name__ == "AsDiscreted":
                discreted_idx = i
                break

        if self.label_indices and discreted_idx >= 0:
            # Insert label selection BEFORE discretization
            transforms.insert(
                discreted_idx,
                SelectLabelsd(keys="pred", label_indices=self.label_indices)
            )
            logger.info(f"Added SelectLabelsd BEFORE AsDiscreted at index {discreted_idx}")

        return transforms

class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")
        self.models = get_bundle_models(app_dir, conf)
        # Add Epistemic model for scoring
        self.epistemic_models = (
            get_bundle_models(app_dir, conf, conf_key="epistemic_model") if conf.get("epistemic_model") else None
        )
        if self.epistemic_models:
            # Get epistemic parameters
            self.epistemic_max_samples = int(conf.get("epistemic_max_samples", "0"))
            self.epistemic_simulation_size = int(conf.get("epistemic_simulation_size", "5"))
            self.epistemic_dropout = float(conf.get("epistemic_dropout", "0.2"))

        self.sam = strtobool(conf.get("sam2", "true"))
        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=f"MONAILabel - Zoo/Bundle ({monailabel.__version__})",
            description="DeepLearning models provided via MONAI Zoo/Bundle",
            version=monailabel.__version__,
        )

    def init_infers(self) -> Dict[str, InferTask]:
        infers: Dict[str, InferTask] = {}
        #################################################
        # Models
        #################################################

        for n, b in self.models.items():
            if "wholeBody_ct_segmentation" in n:
                
                task = CustomBundleInferTask(
                    bundle_path=b,
                    conf=self.conf,
                    label_indices=[0, 1, 2],  # Modify this list as needed
                    type=InferType.SEGMENTATION
                )
                logger.info(f"+++ Custom Inferer for {n} with label filtering")
                infers[n] = task

            if "deepedit" in n:
                # Adding automatic inferer
                i = BundleInferTask(b, self.conf, type="segmentation")
                logger.info(f"+++ Adding Inferer:: {n}_seg => {i}")
                infers[n + "_seg"] = i
                # Adding inferer for managing clicks
                i = BundleInferTask(b, self.conf, type="deepedit")
                logger.info("+++ Adding DeepEdit Inferer")
                infers[n] = i
            else:
                # i = BundleInferTask(b, self.conf)
                # logger.info(f"+++ Adding Inferer:: {n} => {i}")
                # infers[n] = i

                task = CustomBundleInferTask(
                    bundle_path=b,
                    conf=self.conf,
                    label_indices=[0, 1, 2],  # Modify this list as needed
                    type=InferType.SEGMENTATION
                )
                logger.info(f"+++ Custom Inferer for {n} with label filtering")
                infers[n] = task

        #################################################
        # SAM
        #################################################
        if is_sam2_module_available() and self.sam:
            from monailabel.sam2.infer import Sam2InferTask

            infers["sam_2d"] = Sam2InferTask(model_dir=self.model_dir, type=InferType.DEEPGROW, dimension=2)
            infers["sam_3d"] = Sam2InferTask(model_dir=self.model_dir, type=InferType.DEEPGROW, dimension=3)
        return infers

    def init_trainers(self) -> Dict[str, TrainTask]:
        trainers: Dict[str, TrainTask] = {}
        if strtobool(self.conf.get("skip_trainers", "false")):
            return trainers

        for n, b in self.models.items():
            t = BundleTrainTask(b, self.conf)
            if not t or not t.is_valid():
                continue

            logger.info(f"+++ Adding Trainer:: {n} => {t}")
            trainers[n] = t
        return trainers

    def init_strategies(self) -> Dict[str, Strategy]:
        strategies: Dict[str, Strategy] = {
            "random": Random(),
            "first": First(),
        }

        logger.info(f"Active Learning Strategies:: {list(strategies.keys())}")
        return strategies

    def init_scoring_methods(self) -> Dict[str, ScoringMethod]:
        methods: Dict[str, ScoringMethod] = {}
        if not self.conf.get("epistemic_model"):
            return methods

        for n, b in self.epistemic_models.items():
            # Create BundleInferTask task with dropout instantiation for scoring inference
            i = BundleInferTask(
                b,
                self.conf,
                train_mode=True,
                skip_writer=True,
                dropout=self.epistemic_dropout,
                post_filter=[SaveImaged, Invertd],
            )
            methods[n] = EpistemicScoring(
                i, max_samples=self.epistemic_max_samples, simulation_size=self.epistemic_simulation_size
            )
            if not methods:
                continue
            methods = methods if isinstance(methods, dict) else {n: methods[n]}
            logger.info(f"+++ Adding Scoring Method:: {n} => {b}")

        logger.info(f"Active Learning Scoring Methods:: {list(methods.keys())}")
        return methods


"""
Example to run train/infer/scoring task(s) locally without actually running MONAI Label Server
"""


def main():
    import argparse
    import shutil
    from pathlib import Path

    from monailabel.utils.others.generic import device_list, file_ext

    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    home = str(Path.home())
    studies = f"{home}/Datasets/Radiology"

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default=studies)
    parser.add_argument("-m", "--model", default="wholeBody_ct_segmentation")
    parser.add_argument("-t", "--test", default="infer", choices=("train", "infer", "batch_infer"))
    args = parser.parse_args()

    app_dir = os.path.dirname(__file__)
    studies = args.studies
    conf = {
        "models": args.model,
        "preload": "false",
    }

    app = MyApp(app_dir, studies, conf)

    # Infer
    if args.test == "infer":
        sample = app.next_sample(request={"strategy": "first"})
        image_id = sample["id"]
        image_path = sample["path"]

        # Run on all devices
        for device in device_list():
            res = app.infer(request={"model": args.model, "image": image_id, "device": device})
            label = res["file"]
            label_json = res["params"]
            test_dir = os.path.join(args.studies, "test_labels")
            os.makedirs(test_dir, exist_ok=True)

            label_file = os.path.join(test_dir, image_id + file_ext(image_path))
            shutil.move(label, label_file)

            print(label_json)
            with open("test.txt", "w") as f:
                f.write(f"++++ Image File: {image_path}\n")
                f.write(f"++++ Label File: {label_file}\n")
                f.write(f"++++ Device: {device}\n")
                f.write(f"++++ Label JSON: {label_json}\n")
            print(f"++++ Image File: {image_path}")
            print(f"++++ Label File: {label_file}")
            break
        return

    # Batch Infer
    if args.test == "batch_infer":
        app.batch_infer(
            request={
                "model": args.model,
                "multi_gpu": False,
                "save_label": True,
                "label_tag": "original",
                "max_workers": 1,
                "max_batch_size": 0,
            }
        )
        return

    # Train
    app.train(
        request={
            "model": args.model,
            "max_epochs": 10,
            "dataset": "Dataset",  # PersistentDataset, CacheDataset
            "train_batch_size": 1,
            "val_batch_size": 1,
            "multi_gpu": False,
            "val_split": 0.1,
        },
    )


if __name__ == "__main__":
    # export PYTHONPATH=~/Projects/MONAILabel:`pwd`
    # python main.py
    main()

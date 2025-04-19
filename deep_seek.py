import logging
import os
from typing import Dict, List, Optional
from monai.transforms import Invertd, SaveImaged, Transform, MapTransform
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
        # Get original post transforms from bundle config
        transforms = super().post_transforms(data)

        # Add our custom label filtering at the end
        if self.label_indices:
            transforms.append(
                SelectLabelsd(keys="pred", label_indices=self.label_indices)
            )
            logger.info(f"Added SelectLabelsd transform with indices: {self.label_indices}")

        return transforms

class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")
        self.models = get_bundle_models(app_dir, conf)
        self.epistemic_models = (
            get_bundle_models(app_dir, conf, conf_key="epistemic_model") if conf.get("epistemic_model") else None
        )
        if self.epistemic_models:
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

        for n, b in self.models.items():
            # Special handling for whole body CT segmentation
            if n == "wholeBody_ct_segmentation":
                # Specify which labels to keep (0-based indices)
                # Example: keep first 3 labels (adjust according to your label map)
                task = CustomBundleInferTask(
                    bundle_path=b,
                    conf=self.conf,
                    label_indices=[0, 1, 2],  # Modify this list as needed
                    type=InferType.SEGMENTATION
                )
                logger.info(f"+++ Custom Inferer for {n} with label filtering")
                infers[n] = task
            elif "deepedit" in n:
                seg_task = BundleInferTask(b, self.conf, type="segmentation")
                infers[f"{n}_seg"] = seg_task
                deepedit_task = BundleInferTask(b, self.conf, type="deepedit")
                infers[n] = deepedit_task
            else:
                infers[n] = BundleInferTask(b, self.conf)

        if is_sam2_module_available() and self.sam:
            from monailabel.sam2.infer import Sam2InferTask
            infers.update({
                "sam_2d": Sam2InferTask(self.model_dir, InferType.DEEPGROW, 2),
                "sam_3d": Sam2InferTask(self.model_dir, InferType.DEEPGROW, 3)
            })
            
        return infers

    # Rest of the class remains unchanged...
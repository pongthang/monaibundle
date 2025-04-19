import logging
import os
from typing import Dict
from monai.transforms import Invertd, SaveImaged, Transform
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

class SelectLabels(Transform):
    """
    Custom transform to filter specific label indices from segmentation output.
    """
    def __init__(self, label_indices):
        self.label_indices = label_indices

    def __call__(self, data):
        pred = data.get("pred")
        if pred is not None:
            # Select specified channels (assuming channel-first format)
            data["pred"] = pred[self.label_indices, ...]
            
            # Update label names if available in metadata
            label_names = data.get("label_names")
            if label_names:
                data["label_names"] = [label_names[i] for i in self.label_indices]
        return data

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

        # Add inference tasks for each model
        for name, bundle in self.models.items():
            post_filter = []
            
            # Apply label filtering only for whole body CT segmentation
            if name == "wholeBody_ct_segmentation":
                # Example: Keep first 3 labels (modify indices as needed)
                post_filter.append(SelectLabels(label_indices=[0, 1, 2]))
            
            # Create inference task with custom post-processing
            if "deepedit" in name:
                seg_task = BundleInferTask(bundle, self.conf, type="segmentation", post_filter=post_filter)
                infers[f"{name}_seg"] = seg_task
                deepedit_task = BundleInferTask(bundle, self.conf, type="deepedit")
                infers[name] = deepedit_task
            else:
                task = BundleInferTask(bundle, self.conf, post_filter=post_filter)
                infers[name] = task

        # Add SAM inference tasks if enabled
        if is_sam2_module_available() and self.sam:
            from monailabel.sam2.infer import Sam2InferTask
            infers.update({
                "sam_2d": Sam2InferTask(self.model_dir, InferType.DEEPGROW, 2),
                "sam_3d": Sam2InferTask(self.model_dir, InferType.DEEPGROW, 3)
            })
        
        return infers

    # Remaining methods (init_trainers, init_strategies, etc.) unchanged from original
    # ... (include the rest of your existing MyApp class methods here)

# Main execution and other helper functions remain unchanged
# ... (include the rest of your existing main() and other code)
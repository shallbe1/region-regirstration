"""
A two-view sparse feature matching pipeline.

This model contains sub-models for each step:
    feature extraction, feature matching, outlier filtering, pose estimation.
Each step is optional, and the features or matches can be provided as input.
Default: SuperPoint with nearest neighbor matching.

Convention for the matches: m0[i] is the index of the keypoint in image 1
that corresponds to the keypoint i in image 0. m0[i] = -1 if i is unmatched.
"""

from omegaconf import OmegaConf
import time
from . import get_model
from .base_model import BaseModel

to_ctr = OmegaConf.to_container  # convert DictConfig to dict


class TwoViewPipeline(BaseModel):
    default_conf = {
        "extractor": {
            "name": None,
            "trainable": False,
        },
        "matcher": {"name": None},
        "filter": {"name": None},
        "solver": {"name": None},
        "ground_truth": {"name": None},
        "allow_no_extract": False,
        "run_gt_in_forward": False,
    }
    required_data_keys = ["view0", "view1"]
    strict_conf = False  # need to pass new confs to children models
    components = [
        "extractor",
        "matcher",
        "filter",
        "solver",
        "ground_truth",
    ]

    def _init(self, conf):
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(to_ctr(conf.extractor))
            print(conf.extractor.name,1)

        if conf.matcher.name:
            self.matcher = get_model(conf.matcher.name)(to_ctr(conf.matcher))
            print(conf.matcher.name,2)
        if conf.filter.name:
            self.filter = get_model(conf.filter.name)(to_ctr(conf.filter))
            print(conf.filter.name,3)
        if conf.solver.name:
            self.solver = get_model(conf.solver.name)(to_ctr(conf.solver))
            print(conf.solver.name,4)
        if conf.ground_truth.name:
            self.ground_truth = get_model(conf.ground_truth.name)(
                to_ctr(conf.ground_truth)
            )
            print(conf.ground_truth.name,5)
            
            
    def extract_view(self, data, i):
        a2=time.time()
        data_i = data[f"view{i}"]
        pred_i = data_i.get("cache", {})
        skip_extract = len(pred_i) > 0 and self.conf.allow_no_extract
        
        if self.conf.extractor.name and not skip_extract:
            
            pred_i = {**pred_i, **self.extractor(data_i)}
        elif self.conf.extractor.name and not self.conf.allow_no_extract:
            
            pred_i = {**pred_i, **self.extractor({**data_i, **pred_i})}
        a3=time.time()
        #print(a3-a2,"!!!!")
        return pred_i

    def _forward(self, data):
        a0=time.time()
        pred0 = self.extract_view(data, "0")
        pred1 = self.extract_view(data, "1")
        
        pred = {
            **{k + "0": v for k, v in pred0.items()},
            **{k + "1": v for k, v in pred1.items()},
        }

        if self.conf.matcher.name:
            #print(self.matcher,"!!!!!!!!!!!")
            pred = {**pred, **self.matcher({**data, **pred})}
        if self.conf.filter.name:
            pred = {**pred, **self.filter({**data, **pred})}
        if self.conf.solver.name:
            pred = {**pred, **self.solver({**data, **pred})}

        if self.conf.ground_truth.name and self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})
        b0=time.time()
        print(b0-a0,"---------------------")   
        return pred

    def loss(self, pred, data):
        losses = {}
        metrics = {}
        total = 0

        # get labels
        if self.conf.ground_truth.name and not self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

        for k in self.components:
            apply = True
            if "apply_loss" in self.conf[k].keys():
                apply = self.conf[k].apply_loss
            if self.conf[k].name and apply:
                try:
                    losses_, metrics_ = getattr(self, k).loss(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                losses = {**losses, **losses_}
                metrics = {**metrics, **metrics_}
                total = losses_["total"] + total
        return {**losses, "total": total}, metrics

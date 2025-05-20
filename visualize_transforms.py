#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
from itertools import chain

import cv2
import tqdm

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_train_loader,
    DatasetCatalog,
    detection_utils as utils,
    MetadataCatalog,
)
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from maskdino import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskdino_config,
    DetrDatasetMapper,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler

from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer
)
from detectron2.data.datasets import register_coco_instances
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def register_custom_coco_dataset(dataset_path: str = "/h/jquinto/MaskDINO/datasets/lifeplan/") -> None:
   annotations_path = dataset_path + "annotations/"
   register_coco_instances(
       "lifeplan_train",
       {},
       annotations_path + "instances_train2017.json",
       dataset_path + "train2017",
   )
   register_coco_instances(
       "lifeplan_valid",
       {},
       annotations_path + "instances_val2017.json",
       dataset_path + "val2017",
   )
   
def setup(args):
    """
    Create configs and perform basic setups.
    """
    register_custom_coco_dataset()
    cfg = get_cfg()
    cfg.DATASETS.TRAIN = ("lifeplan_train",)
    cfg.DATASETS.TEST = ("lifeplan_valid",)
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument(
        "--source",
        choices=["annotation", "dataloader"],
        required=True,
        help="visualize the annotations or the data loader (with pre-processing)",
    )
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


def main() -> None:
    global img
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def output(vis, fname):
        if args.show:
            print(fname)
            cv2.imshow("window", vis.get_image()[:, :, ::-1])
            cv2.waitKey()
        else:
            filepath = os.path.join(dirname, fname)
            print("Saving to {} ...".format(filepath))
            vis.save(filepath)

    scale = 1.0
    if args.source == "dataloader":
        mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
        train_data_loader = build_detection_train_loader(cfg, mapper=mapper)
        for batch in train_data_loader:
            for per_image in batch:
                # Pytorch tensor is in (C, H, W) format
                img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
                img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

                visualizer = Visualizer(img, metadata=metadata, scale=scale)
                target_fields = per_image["instances"].get_fields()
                labels = [
                    metadata.thing_classes[i] for i in target_fields["gt_classes"]
                ]
                vis = visualizer.overlay_instances(
                    labels=labels,
                    boxes=target_fields.get("gt_boxes", None),
                    masks=target_fields.get("gt_masks", None),
                    keypoints=target_fields.get("gt_keypoints", None),
                )
                output(vis, str(per_image["image_id"]) + ".jpg")
    else:
        dicts = list(
            chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN])
        )
        if cfg.MODEL.KEYPOINT_ON:
            dicts = filter_images_with_few_keypoints(dicts, 1)
        for dic in tqdm.tqdm(dicts):
            img = utils.read_image(dic["file_name"], "RGB")
            visualizer = Visualizer(img, metadata=metadata, scale=scale)
            vis = visualizer.draw_dataset_dict(dic)
            output(vis, os.path.basename(dic["file_name"]))


if __name__ == "__main__":
    main()  # pragma: no cover
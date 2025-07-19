# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li.
import copy
import logging

import numpy as np
from numpy import random
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances, PolygonMasks

from pycocotools import mask as coco_mask
from PIL import Image
from fvcore.transforms.transform import (
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    PadTransform,
    Transform,
    TransformList,
    VFlipTransform,
)
from detectron2.data.transforms.transform import ResizeTransform
from detectron2.structures import Boxes, pairwise_iou



__all__ = ["COCOInstanceNewBaselineDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class ResizeScaleLog(T.Augmentation):
    """
    Takes target size as input and randomly scales the given target size between `min_scale`
    and `max_scale`. It then scales the input image such that it fits inside the scaled target
    box, keeping the aspect ratio constant.
    This implements the resize part of the Google's 'resize_and_crop' data augmentation:
    https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/input_utils.py#L127
    """

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        target_height: int,
        target_width: int,
        interp: int = Image.BILINEAR,
    ):
        """
        Args:
            min_scale: minimum image scale range.
            max_scale: maximum image scale range.
            target_height: target image height.
            target_width: target image width.
            interp: image interpolation method.
        """
        super().__init__()
        self._init(locals())

    def _get_resize(self, image: np.ndarray, scale: float) -> Transform:
        input_size = image.shape[:2]

        # Compute new target size given a scale.
        target_size = (self.target_height, self.target_width)
        target_scale_size = np.multiply(target_size, scale)

        # Compute actual rescaling applied to input image and output size.
        output_scale = np.minimum(
            target_scale_size[0] / input_size[0], target_scale_size[1] / input_size[1]
        )
        output_size = np.round(np.multiply(input_size, output_scale)).astype(int)

        return ResizeTransform(
            input_size[0], input_size[1], int(output_size[0]), int(output_size[1]), self.interp
        )

    def get_transform(self, image: np.ndarray) -> Transform:
        # random_scale = np.random.lognormal(mean=1.0, sigma=0.5)
        # random_scale = np.random.uniform(self.min_scale, self.max_scale)
        random_scale = np.random.uniform(np.log(self.min_scale), np.log(self.max_scale)) # Log-uniform
        return self._get_resize(image, np.exp(random_scale))
    
class ResizeScale(Augmentation):
    """
    Takes target size as input and randomly scales the given target size between `min_scale`
    and `max_scale`. It then scales the input image such that it fits inside the scaled target
    box, keeping the aspect ratio constant. If the resized image contains any "cut-off" bounding boxes,
    it samples again until no "cut-off" bounding boxes are present.
    """

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        target_height: int,
        target_width: int,
        interp: int = Image.BILINEAR,
        max_trials: int = 1000,
    ):
        """
        Args:
            min_scale: minimum image scale range.
            max_scale: maximum image scale range.
            target_height: target image height.
            target_width: target image width.
            interp: image interpolation method.
            max_trials: maximum number of trials to find a valid transformation.
        """
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_height = target_height
        self.target_width = target_width
        self.interp = interp
        self.max_trials = max_trials

    def _get_resize(self, image, scale):
        input_size = image.shape[:2]

        # Compute new target size given a scale.
        target_size = (self.target_height, self.target_width)
        target_scale_size = np.multiply(target_size, scale)

        # Compute actual rescaling applied to input image and output size.
        output_scale = np.minimum(
            target_scale_size[0] / input_size[0], target_scale_size[1] / input_size[1]
        )
        output_size = np.round(np.multiply(input_size, output_scale)).astype(int)

        return output_scale, output_size

    def are_boxes_fully_contained(self, boxes, output_size):
        output_height, output_width = output_size
        return np.all(boxes[:, 0] >= 0) and \
               np.all(boxes[:, 1] >= 0) and \
               np.all(boxes[:, 2] <= output_width) and \
               np.all(boxes[:, 3] <= output_height)

    def get_transform(self, image, boxes):
        if boxes is None or len(boxes) == 0:
            return NoOpTransform()
        
        input_size = image.shape[:2]
        
        for _ in range(self.max_trials):
            random_scale = np.random.uniform(self.min_scale, self.max_scale)
            output_scale, output_size = self._get_resize(image, random_scale)

            # Adjust the bounding boxes
            scaled_boxes = boxes * output_scale

            if self.are_boxes_fully_contained(scaled_boxes, output_size):
                return ResizeTransform(
                    input_size[1], input_size[0], int(output_size[1]), int(output_size[0]), self.interp
                )
        
        # If no valid transform found, return NoOpTransform
        return NoOpTransform()


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        # MinIoURandomCrop(),
        ResizeScaleAndCrop(min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation

class COCOInstanceNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOInstanceNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # Get the bounding boxes from annotations
        if "annotations" in dataset_dict:
            boxes = np.array([obj["bbox"] for obj in dataset_dict["annotations"]], dtype=np.float32)
            # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
            boxes[:, 2:] += boxes[:, :2]
        else:
            boxes = np.array([])

        aug_input = T.AugInput(image, boxes=boxes)

        transforms = T.AugmentationList(self.tfm_gens)(aug_input)
        image = aug_input.image
        boxes = aug_input.boxes

        # padding_mask = np.ones(image.shape[:2])
        # padding_mask = transforms.apply_segmentation(padding_mask)
        # padding_mask = ~padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            instances = utils.annotations_to_instances(annos, image_shape)
            if not instances.has('gt_masks'):
                instances.gt_masks = PolygonMasks([])
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = utils.filter_empty_instances(instances)

            h, w = instances.image_size
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks

            dataset_dict["instances"] = instances

        return dataset_dict
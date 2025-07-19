import json
import argparse
from sahi import AutoDetectionModel
from sahi.predict import predict
import torch
import numpy as np
from sahi.scripts.coco_evaluation import evaluate
import os

def main(args):
    assert torch.cuda.is_available()
    # Define default values for arguments
    config_path = args.config_path 
    crop_fac = args.crop_fac or 16
    postprocess_match_threshold = args.postprocess_match_threshold or 0.3

    # Define image shape and cropping dimensions
    scale_factor = args.scale_factor
    im_shape_0 = 5464 * scale_factor
    im_shape_1 = 8192 * scale_factor 
    
    if (args.slice_height is None and args.slice_width is None) and args.crop_fac is not None:
        crop_rows = im_shape_0 // crop_fac
        crop_cols = im_shape_1 // crop_fac
    elif args.slice_height is not None and args.slice_width is not None:
        if args.super_resolution:
            crop_rows = args.slice_height * scale_factor
            crop_cols = args.slice_width * scale_factor
        else: 
            crop_rows = args.slice_height
            crop_cols = args.slice_width
    print(f"crop_rows: {crop_rows}")
    print(f"crop_cols: {crop_cols}")
    print(f"overlap: {args.overlap}")
        
    # Set device for model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load detection model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='detectron2',
        model_path=args.model_path,
        config_path=config_path,
        confidence_threshold=args.model_confidence_threshold,
        # image_size=1024,
        device=device,
    )
    detection_model.load_model()

    # Perform prediction
    postprocess_match_threshold_id = str(args.postprocess_match_threshold).replace('.', '')
    model_conf_threshold = str(args.model_confidence_threshold).replace('.', '')
    # model_ids = args.model_path.split('/')[4].split('_')
    # model_id = "_".join(model_ids[1:])
    # exp_name = f"exp_{postprocess_match_threshold_id}_{model_conf_threshold}_conf_{model_id}"

    model_ids = args.model_path.split('/')[4].split('_')
    model_id_from_folder = "_".join(model_ids[1:])
    model_id = args.exp_name or model_id_from_folder
    exp_name = f"exp_{postprocess_match_threshold_id}_{model_conf_threshold}_conf_{model_id}"

    if os.path.exists(f"/h/jquinto/MaskDINO/runs/predict/{exp_name}"):
        exp_name = f"{exp_name}_current" 
    print(exp_name)

    if args.predict:
        predict(
            detection_model=detection_model,
            model_confidence_threshold=args.model_confidence_threshold,
            model_device=device,
            model_category_mapping={"1": "b"},
            source=args.dataset_img_path,
            no_standard_prediction=True,
            # image_size=args.image_size,
            slice_height=crop_rows,
            slice_width=crop_cols,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            postprocess_type="GREEDYNMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=postprocess_match_threshold,
            postprocess_class_agnostic=True,
            export_pickle=False,
            export_crop=False, ## SET to FALSE to conserve space
            novisual=True, ## SET to TRUE to MAKE FASTER
            dataset_json_path=args.dataset_json_path,
            visual_hide_labels=True,
            visual_hide_conf=True,
            name = exp_name,
        )

    # Load the JSON data
    with open(args.dataset_json_path, "r") as f:
        json_data = json.load(f)

    # Update category_id in annotations
    for annotation in json_data['annotations']:
        if annotation['category_id'] == 1:
            annotation['category_id'] = 0

    # Update the categories key
    for category in json_data['categories']:
        if category['id'] == 1:
            category['id'] = 0

    # Save updated JSON
    updated_json_path = f"{args.dataset_json_path[:-5]}_updated.json"
    with open(updated_json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    # If super resolution is enabled, divide the coordinates of the predictions by 4 and areas by 4**2
    if args.super_resolution is True and scale_factor > 1:
        # Rescale the predictions from the JSON file 
        print(f"SUPER RESOLUTION ENABLED: SCALING DOWN PREDICTIONS BY FACTOR {scale_factor}")
        with open(f"/h/jquinto/MaskDINO/runs/predict/{exp_name}/result.json", "r") as f:
            SR_preds = json.load(f)
    
        for pred in SR_preds:
            bbox = pred['bbox'].copy()
            bbox = [i/scale_factor for i in bbox]
            pred['bbox'] = bbox
            
            seg = pred['segmentation'].copy()
            for i in range(len(seg)):
                for j in range(len(seg[i])):
                    seg[i][j] /= scale_factor
        
            pred['area'] /= scale_factor**2
        
        with open(f"/h/jquinto/MaskDINO/runs/predict/{exp_name}/result.json", "w") as f:
            json.dump(SR_preds, f, indent=4)

    # Evaluation should ALWAYS be the same as ORIGINAL SCALE
    areas = [144, 1024, 10000000000] 
    
    # Perform COCO evaluation
    for det_type in ["bbox", "segm"]:
        evaluate(
            dataset_json_path=updated_json_path,
            result_json_path=f"/h/jquinto/MaskDINO/runs/predict/{exp_name}/result.json",
            out_dir=f"/h/jquinto/MaskDINO/runs/predict/{exp_name}",
            type=det_type,
            classwise=True,
            max_detections=1000,
            areas = areas
        )
    """
    cocoEval.params.areaRng = [
                [0**2, areas[2]],
                [0**2, areas[0]],
                [areas[0], areas[1]],
                [areas[1], areas[2]],
            ]

evaluate(
    dataset_json_path="/h/jquinto/lifeplan_b_v9_cropped_center/annotations/instances_val2017_updated.json", 
    result_json_path="/h/jquinto/merged_predictions_MaskRCNN_512.json", 
    out_dir="/h/jquinto/MaskDINO/runs/predict/sahi_512_merged",
    type="segm", 
    max_detections=1000,
    classwise=True,
    areas=[144, 1024, 10000000000] 
)
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAHI prediction and COCO evaluation.")
    parser.add_argument("--model_path", type=str, help="Path to the model file.")
    parser.add_argument("--exp_name", type=str, default = None, help="Experiment Name")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--dataset_json_path", type=str, help="Path to the dataset JSON file.")
    parser.add_argument("--dataset_img_path", type=str, help="Path to the dataset image file(s).")
    parser.add_argument("--config_path", type=str, help="Path to the config file.")
    parser.add_argument("--crop_fac", type=int, help="Crop factor for slicing the image.", default=16)
    parser.add_argument("--postprocess_match_threshold", type=float, help="Post-process match threshold.", default=0.5)
    parser.add_argument("--model_confidence_threshold", type=float, help="Model Confidence Threshold.", default=0.25)
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument("--super_resolution", action="store_true")
    parser.add_argument("--slice_height", type=int, default=None)
    parser.add_argument("--slice_width", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=0.4)

    args = parser.parse_args()
    main(args)
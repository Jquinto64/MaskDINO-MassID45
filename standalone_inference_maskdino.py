"""
Standalone inference script for MaskDINO models.
Runs on a single image or a directory of images.
Visualizes predictions and saves them to an output directory.
"""
import logging
import argparse
import os
import glob
import sys
import cv2
import numpy as np
import torch

# --- 1. SETUP PATHS ---
# Keep the specific path insertion from your snippet
sys.path.insert(0, ".")

# --- 2. IMPORTS ---
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import Instances

# MaskDINO imports
try:
    from maskdino import add_maskdino_config
    # Explicitly import modeling if required by the config registry, 
    # though usually add_maskdino_config + factory is enough.
    from maskdino.modeling import * 
except ImportError as e:
    print(f"Error importing MaskDINO: {e}")
    print("Please ensure your 'sys.path.insert' points to the correct MaskDINO repository.")
    sys.exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def setup_predictor(args):
    """
    Sets up the Detectron2/MaskDINO config and predictor.
    """
    logger.info("Setting up configuration...")
    cfg = get_cfg()
    cfg.set_new_allowed(True) 
    
    # Add project specific configs
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    
    # Merge file config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    cfg.merge_from_file(args.config)
    
    # Load Weights
    cfg.MODEL.WEIGHTS = args.model_path
    
    # Set Device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set Inference Image Size
    # DefaultPredictor will resize input images to this size, run inference, 
    # and then scale predictions back to original image size automatically.
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024
    
    cfg.freeze()

    logger.info(f"Loading model from {args.model_path}...")
    predictor = DefaultPredictor(cfg)
    print(predictor.cfg.INPUT.IMAGE_SIZE)
    return predictor, cfg

def predict_on_single_image(predictor, img_path, debug=False):
    """
    Runs inference on a single image.
    """
    # Read image using OpenCV
    original_image = cv2.imread(img_path)
    if original_image is None:
        logger.error(f"Could not read image: {img_path}")
        return None, None

    # Run Inference
    # DefaultPredictor handles:
    # 1. Resizing input -> 1024
    # 2. Forward pass
    # 3. Resizing outputs -> Original Size
    outputs = predictor(original_image)

    if debug:
        print("\n--- DEBUG INFO ---")
        instances = outputs['instances']
        print(f"Detected {len(instances)} objects")
        if len(instances) > 0:
            if instances.has("pred_masks"):
                sample_preds = instances.pred_masks.cpu().detach().numpy()
                print(f"Mask Shape: {sample_preds.shape} (Should match original image {original_image.shape[:2]})")

    return outputs, original_image

def display_predictions(image, predictions, output_path, metadata):
    """
    Visualizes predictions and saves the result to disk.
    """
    if predictions is None:
        return

    # Convert BGR (OpenCV) to RGB for Visualizer
    visualizer = Visualizer(
        image[:, :, ::-1], 
        metadata=metadata, 
        scale=1.0, 
        instance_mode=ColorMode.IMAGE
    )
    
    # Draw predictions
    instances = predictions["instances"].to("cpu")
    instances_ = Instances(instances.image_size)# <class 'detectron2.structures.instances.Instances'>
    flag = False
    for index in range(len(instances)):
        # print(instances[index].scores)
        score = instances[index].scores[0]
        if score > 0.25: # confidence score
            if flag == False:
                instances_ = instances[index]
                flag = True
            else:
                instances_ = Instances.cat([instances_, instances[index]])
    vis_output = visualizer.draw_instance_predictions(predictions=instances_)

    # Get the result image (in RGB) and convert back to BGR for OpenCV saving
    result_image = vis_output.get_image()[:, :, ::-1]

    # Save
    cv2.imwrite(output_path, result_image)
    logger.info(f"Saved visualization to: {output_path}")

def main(args):
    # 1. Setup Predictor
    predictor, cfg = setup_predictor(args)

    # 2. Setup Metadata for Visualization
    # Define your specific mapping here
    category_mapping = {"1": "b"} 
    
    # Create metadata object
    metadata = MetadataCatalog.get("__unused_maskdino_inference__") 
    metadata.thing_classes = list(category_mapping.values())
    
    # 3. Handle Input Directory or File
    input_path = args.imgs_directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    image_files = []
    if os.path.isdir(input_path):
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
            image_files.extend(glob.glob(os.path.join(input_path, ext.upper())))
    elif os.path.isfile(input_path):
        image_files = [input_path]
    else:
        logger.error(f"Input path {input_path} is not valid.")
        return

    logger.info(f"Found {len(image_files)} images to process.")

    # 4. Run Inference Loop
    for i, img_file in enumerate(image_files):
        filename = os.path.basename(img_file)
        save_path = os.path.join(output_dir, f"pred_{filename}")

        logger.info(f"[{i+1}/{len(image_files)}] Processing {filename}...")
        
        predictions, original_img = predict_on_single_image(
            predictor, 
            img_file, 
            debug=args.debug
        )

        if predictions is not None:
            display_predictions(original_img, predictions, save_path, metadata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone inference script for MaskDINO")
    
    # Path Arguments
    parser.add_argument("--model_path", required=True, help="Path to the model .pth file")
    parser.add_argument("--config", required=True, help="Path to the MaskDINO .yaml config file")
    parser.add_argument("--imgs_directory", required=True, help="Path to input image or directory of images")
    parser.add_argument("--output_dir", default="output_predictions_maskdino", help="Directory to save visualized outputs")
    
    # Optional Arguments
    parser.add_argument("--debug", action="store_true", help="Print debug information about masks/boxes")
    
    args = parser.parse_args()
    
    main(args)
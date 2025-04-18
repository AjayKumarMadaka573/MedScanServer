import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import sys
import torch.nn.functional as F
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Configure output encoding
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
])

domain_tasks = {
    "bcn_vs_ham_age_u30": "bcn_age_u30",
    "msk_vs_bcn_age_u30": "msk_age_u30",
    "bcn_vs_ham_loc_head_neck": "bcn_loc_head_neck",
    "bcn_vs_msk_headloc": "bcn_loc_head_neck",
    "ham_vs_msk_loc_head_neck": "ham_loc_head_neck",
    "ham_age_u30vsmsk_age_u30": "ham_age_u30",
    "bcn_vs_ham_loc_palms_soles": "bcn_loc_palms_soles"
}

class Classifier(nn.Module):
    def __init__(self, backbone, num_classes=2, bottleneck_dim=512):
        super().__init__()
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.bottleneck(x)
        return self.classifier(features), features

def log_error(context, error, details=None):
    error_data = {
        "context": context,
        "error": str(error),
        "details": details
    }
    logging.error(json.dumps(error_data, indent=2))
    return error_data

def load_model(task_path):
    try:
        # Verify file exists before attempting to load
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"Model checkpoint not found at {task_path}")
            
        backbone = models.resnet18(weights=None)
        model = Classifier(backbone)

        # Load checkpoint with explicit error handling
        checkpoint = torch.load(task_path, map_location=device, weights_only=False)
        
        # Handle 'module.' prefix for multi-GPU saved models
        if any(k.startswith('module.') for k in checkpoint.keys()):
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        
        model.load_state_dict(checkpoint, strict=False)
        return model.to(device).eval()
        
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

def process_image(image_path):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}")
            
        img = Image.open(image_path).convert('RGB')
        return transform(img).unsqueeze(0).to(device)
    except Exception as e:
        raise RuntimeError(f"Image processing failed: {str(e)}")

def validate_paths(base_dir, subdirs=None):
    """Validate that required directories exist"""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    if subdirs:
        for subdir in subdirs:
            path = os.path.join(base_dir, subdir)
            if not os.path.exists(path):
                logging.warning(f"Subdirectory not found: {path}")

def get_relative_path(*path_parts):
    """Get path relative to the script directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, *path_parts)

def main():
    try:
        # Validate input
        if len(sys.argv) < 2:
            raise ValueError("Missing required argument: image_path")
            
        image_path = sys.argv[1]
        
        # Set up paths - using relative paths from script location
        base_dir = get_relative_path()
        model_dir = get_relative_path("model_checkpoints")
        feat_dir = get_relative_path("domain_features_dann")
        
        # Validate directories exist
        validate_paths(model_dir)
        validate_paths(feat_dir)
        
        # Initialize result tracking
        best_result = {
            "task": None, 
            "distance": float('inf'), 
            "label": None,
            "confidence": None,
            "warnings": []
        }
        
        # Process each task
        for task, task_name in domain_tasks.items():
            try:
                # Construct paths
                model_path = os.path.join(model_dir, f"{task}.log", "checkpoints", "best.pth")
                feat_path = os.path.join(feat_dir, f"{task}_features.npy")
                
                # Skip if required files don't exist
                if not all(os.path.exists(p) for p in [model_path, feat_path]):
                    missing = [p for p in [model_path, feat_path] if not os.path.exists(p)]
                    best_result["warnings"].append(f"Missing files for task {task}: {missing}")
                    continue
                
                # Load and process
                model = load_model(model_path)
                img_tensor = process_image(image_path)
                
                with torch.no_grad():
                    logits, features = model(img_tensor)
                    probs = F.softmax(logits, dim=1) 
                    label = int(torch.argmax(logits).item())
                    confidence = float(probs[0][label].item())
                    features = features.cpu().numpy().squeeze()
                    features /= np.linalg.norm(features)
                
                domain_feats = np.load(feat_path)
                domain_feats = domain_feats / np.linalg.norm(domain_feats, axis=1, keepdims=True)
                distance = float(np.mean(np.linalg.norm(domain_feats - features, axis=1)))
                
                if distance < best_result["distance"]:
                    best_result.update({
                        "task": task_name,
                        "distance": distance,
                        "label": label,
                        "confidence": confidence
                    })
                    
            except Exception as e:
                error_details = {
                    "task": task,
                    "error": str(e),
                    "type": type(e).__name__
                }
                best_result["warnings"].append(error_details)
                logging.error(f"Task {task} failed: {str(e)}")
                continue
        
        # Prepare final output
        output = {
            "status": "success" if best_result["task"] else "partial_success",
            "result": {
                "prediction": best_result["label"],
                "confidence": best_result["confidence"],
                "task": best_result["task"],
                "distance": best_result["distance"]
            },
            "metadata": {
                "processed_tasks": len(domain_tasks) - len(best_result["warnings"]),
                "failed_tasks": len(best_result["warnings"])
            }
        }
        
        if best_result["warnings"]:
            output["warnings"] = best_result["warnings"]
        
        # Convert label to human-readable
        if output["result"]["prediction"] is not None:
            label_map = {0: "Benign", 1: "Malignant"}
            output["result"]["prediction"] = label_map.get(output["result"]["prediction"], "Unknown")
        
        print(json.dumps(output, indent=2))
        
    except Exception as e:
        error_output = {
            "status": "error",
            "error": str(e),
            "type": type(e).__name__,
            "details": {
                "image_path": image_path if 'image_path' in locals() else None,
                "python_version": sys.version,
                "torch_version": torch.__version__
            }
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
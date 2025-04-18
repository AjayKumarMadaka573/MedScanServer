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

# Configure logging to only go to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)  # Only log to stderr
    ]
)

# Ensure clean stdout for JSON output only
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_print_json(data):
    """Safely print JSON to stdout without any other output"""
    json.dump(data, sys.stdout, separators=(',', ':'))
    sys.stdout.write('\n')
    sys.stdout.flush()

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

def load_model(task_path):
    try:
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"Model checkpoint not found at {task_path}")
            
        backbone = models.resnet18(weights=None)
        model = Classifier(backbone)

        checkpoint = torch.load(task_path, map_location=device)
        
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

def main():
    # Initialize default output structure
    output = {
        "status": "error",
        "error": "Unknown error",
        "result": None,
        "metadata": {
            "processed_tasks": 0,
            "failed_tasks": len(domain_tasks)
        }
    }

    try:
        if len(sys.argv) < 2:
            raise ValueError("Missing required argument: image_path")
            
        image_path = sys.argv[1]
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "model_checkpoints")
        feat_dir = os.path.join(base_dir, "domain_features_dann")

        best_result = {
            "task": None, 
            "distance": float('inf'), 
            "label": None,
            "confidence": None,
            "warnings": []
        }

        processed_tasks = 0
        
        for task, task_name in domain_tasks.items():
            try:
                model_path = os.path.join(model_dir, f"{task}.log", "checkpoints", "best.pth")
                feat_path = os.path.join(feat_dir, f"{task}_features.npy")
                
                if not os.path.exists(model_path):
                    best_result["warnings"].append(f"Model not found: {model_path}")
                    continue
                if not os.path.exists(feat_path):
                    best_result["warnings"].append(f"Features not found: {feat_path}")
                    continue

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
                
                processed_tasks += 1
                    
            except Exception as e:
                best_result["warnings"].append({
                    "task": task,
                    "error": str(e),
                    "type": type(e).__name__
                })
                continue

        # Prepare final output
        if best_result["task"] is not None:
            output["status"] = "success" if not best_result["warnings"] else "partial_success"
            output["result"] = {
                "prediction": "Benign" if best_result["label"] == 0 else "Malignant",
                "confidence": best_result["confidence"],
                "task": best_result["task"],
                "distance": best_result["distance"]
            }
            output["metadata"]["processed_tasks"] = processed_tasks
            output["metadata"]["failed_tasks"] = len(best_result["warnings"])
            
            if best_result["warnings"]:
                output["warnings"] = best_result["warnings"]
        
    except Exception as e:
        output["error"] = str(e)
        output["details"] = {
            "type": type(e).__name__,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__
        }
    finally:
        # Ensure only clean JSON is printed to stdout
        safe_print_json(output)
        sys.exit(0 if output["status"] in ["success", "partial_success"] else 1)

if __name__ == "__main__":
    # Suppress all warnings to prevent stdout pollution
    import warnings
    warnings.filterwarnings("ignore")
    main()
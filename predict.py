import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import sys
import torch.nn.functional as F
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

# def load_model(task_path):
#     backbone = models.resnet18(weights=None)
#     model = Classifier(backbone)
    
#     try:
#         checkpoint = torch.load(task_path, map_location=device, weights_only=False)
#     except Exception as e:
#         print(f"Error loading checkpoint file: {str(e)}", file=sys.stderr)
#         raise RuntimeError("Checkpoint loading failed. Ensure the checkpoint is compatible with your PyTorch version.")
    
#     # Check if the checkpoint keys contain 'module.' and remove it for multi-GPU model saving
#     if any(k.startswith('module.') for k in checkpoint.keys()):
#         checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    
#     # Try loading the checkpoint into the model
#     try:
#         model.load_state_dict(checkpoint, strict=False)
#     except RuntimeError as e:
#         print(f"Error loading state_dict: {str(e)}", file=sys.stderr)
#         raise RuntimeError("Mismatch between model architecture and checkpoint. Ensure they are compatible.")
    
#     return model.to(device).eval()

def load_model(task_path):
    """Load a PyTorch model from checkpoint with robust error handling."""
    try:
        # Initialize model
        backbone = models.resnet18(weights=None)
        model = Classifier(backbone)
        
        # Load checkpoint
        checkpoint = torch.load(task_path, map_location=device, weights_only=False)
        
        # Try loading checkpoint directly first
        try:
            model.load_state_dict(checkpoint, strict=False)
        except RuntimeError:
            # If direct loading fails, try removing 'module.' prefix from keys
            new_checkpoint = {}
            for key in checkpoint.keys():
                new_key = key.replace('module.', '')
                new_checkpoint[new_key] = checkpoint[key]
            model.load_state_dict(new_checkpoint, strict=False)
        
        # Move model to device and set to eval mode
        model = model.to(device).eval()
        return model
        
    except Exception as e:
        error_msg = f"Failed to load model from {task_path}:\n{str(e)}"
        if isinstance(e, RuntimeError) and "Missing key(s)" in str(e):
            error_msg += "\nPossible model architecture mismatch"
        elif isinstance(e, FileNotFoundError):
            error_msg += "\nCheck if file exists and path is correct"
        raise RuntimeError(error_msg)
    
def process_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        return transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Image processing error: {str(e)}", file=sys.stderr)
        return None

def convert_to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    return obj

def main():
    try:
        if len(sys.argv) < 2:
            raise ValueError("Missing image path argument")
            
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        model_dir = os.path.join(os.path.dirname(__file__), "model_checkpoints")
        feat_dir = os.path.join(os.path.dirname(__file__), "domain_features_dann")
        
        best_result = {"task": None, "distance": float('inf'), "label": None}
        
        for task in domain_tasks:
            try:
                model_path = os.path.join(model_dir, f"{task}.log", "checkpoints", "best.pth")
                feat_path = os.path.join(feat_dir, f"{task}_features.npy")
                
                if not all(os.path.exists(p) for p in [model_path, feat_path]):
                    
                    continue
                
                model = load_model(model_path)
                img_tensor = process_image(image_path)
                if img_tensor is None:
                    continue
                
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
                        "task": task,
                        "distance": distance,
                        "label": label,
                        "confidence": confidence
                    })
                    
            except Exception as e:
                print(f"Error processing {task}: {str(e)}", file=sys.stderr)
                continue
        
        if best_result["task"] is None:
            raise RuntimeError("No valid tasks processed")
        best_result["task"] = domain_tasks[best_result["task"]]
        label_map = {0: "Benign", 1: "Malignant"}
        best_result["label"] = label_map.get(best_result["label"], "Unknown")
        # best_result["confidence"] = round(float(best_result["confidence"]) * 100, 2)  

        # Ensure all values are JSON serializable
        serializable_result = convert_to_serializable(best_result)
        print(json.dumps(serializable_result))
        
    except Exception as e:
        error_msg = {"error": str(e)}
        print(json.dumps(error_msg), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
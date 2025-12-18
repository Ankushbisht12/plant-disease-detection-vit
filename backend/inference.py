import torch
import timm
from PIL import Image
import torchvision.transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "ML/saved_model/vit_model.pth"
CLASS_NAMES_PATH = "ML/class_names.txt"

# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model():
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=len(CLASS_NAMES)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(model, image: Image.Image):
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    class_name = CLASS_NAMES[pred_idx.item()]

    crop = class_name.split("___")[0]
    disease = class_name.split("___")[1]

    print("Predicted index:", pred_idx)
    print("Class name:", class_name)
    print("Confidence:", confidence)


    return {
        "crop": crop,
        "disease": disease,
        "confidence": float(confidence.item())
    }

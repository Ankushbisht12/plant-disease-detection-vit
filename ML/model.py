# ml/model.py

import timm

def build_vit_model(num_classes):
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes
    )

    # 🔒 Freeze backbone (VERY IMPORTANT)
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False

    return model

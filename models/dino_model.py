import torch
from torchvision import transforms
from PIL import Image

class DINOModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16", pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode_image(self, image: Image.Image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image_tensor)
        return features.flatten(1) / features.norm(dim=-1, keepdim=True)
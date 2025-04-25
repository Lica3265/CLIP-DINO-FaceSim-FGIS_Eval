import torch
import clip
from PIL import Image

class CLIPModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    def encode_text(self, text: str):
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def encode_image(self, image: Image.Image):
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        return image_features / image_features.norm(dim=-1, keepdim=True)
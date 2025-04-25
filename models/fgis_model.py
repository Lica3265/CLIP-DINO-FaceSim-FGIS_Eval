import torch
from PIL import Image

class FGISModel:
    def __init__(self, device='cpu'):
        self.device = device

    def encode_image(self, image: Image.Image):
        # 細粒度特徵提取（此處用隨機特徵模擬）
        return torch.randn(1, 512).to(self.device) / torch.norm(torch.randn(1, 512).to(self.device))
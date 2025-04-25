import torch
from PIL import Image

class FaceSimModel:
    def __init__(self, device='cpu'):
        self.device = device

    def encode_face(self, image: Image.Image):
        # 人臉特徵提取（此處用隨機特徵模擬）
        return torch.randn(1, 512).to(self.device) / torch.norm(torch.randn(1, 512).to(self.device))
# model_factory.py
from models.clip_model import CLIPModel
from models.dino_model import DINOModel
from models.facesim_model import FaceSimModel
from models.fgis_model import FGISModel

class ModelFactory:
   
    @staticmethod
    def create_model(model_type, device):
        models = {
            "CLIP": CLIPModel,
            "DINO": DINOModel,
            "FaceSim": FaceSimModel,
            "FGIS": FGISModel
        }
        if model_type in models:
            return models[model_type](device)
        else:
            raise ValueError(f"unknown: {model_type}")
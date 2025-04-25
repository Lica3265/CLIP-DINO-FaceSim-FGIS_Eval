import torch
from PIL import Image
from models.clip_model import CLIPModel
from models.dino_model import DINOModel
from models.facesim_model import FaceSimModel
from models.fgis_model import FGISModel
from metrics.similarity_metrics import (
    evaluate_clip_text_image_similarity,
    evaluate_clip_image_similarity,
    evaluate_dino_image_similarity,
    evaluate_facesim_similarity,
    evaluate_fgis_similarity
)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化各模型
    clip_model = CLIPModel(device)
    dino_model = DINOModel(device)
    facesim_model = FaceSimModel(device)
    fgis_model = FGISModel(device)

    # 測試數據
    text_input = "A dog running in the park."
    image_path1 = "data/image1.jpg"
    image_path2 = "data/image2.jpg"
    image1 = Image.open(image_path1).convert('RGB')
    image2 = Image.open(image_path2).convert('RGB')

    clip_score = evaluate_clip_text_image_similarity(clip_model, text_input, image1)
    clip_image_score = evaluate_clip_image_similarity(clip_model, image1, image2)
    dino_score = evaluate_dino_image_similarity(dino_model, image1, image2)
    facesim_score = evaluate_facesim_similarity(facesim_model, image1, image2)
    fgis_score = evaluate_fgis_similarity(fgis_model, image1, image2)

    print(f"CLIP-T Text-Image Similarity: {clip_score * 100:.2f}%")
    print(f"CLIP-I Text-Image Similarity: {clip_image_score * 100:.2f}%")
    print(f"DINO Image-Image Similarity: {dino_score * 100:.2f}%")
    print(f"FaceSim Similarity: {facesim_score * 100:.2f}%")
    print(f"FGIS Similarity: {fgis_score * 100:.2f}%")
    
if __name__ == "__main__":
    main()
import torch.nn.functional as F

def cosine_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2).item()

def evaluate_clip_text_image_similarity(clip_model, text, image):
    text_features = clip_model.encode_text(text)
    image_features = clip_model.encode_image(image)
    return cosine_similarity(text_features, image_features)

def evaluate_clip_image_similarity(clip_model, image1, image2):
    image1_features = clip_model.encode_image(image1)
    image2_features = clip_model.encode_image(image2)
    return cosine_similarity(image1_features, image2_features)

def evaluate_dino_image_similarity(dino_model, image1, image2):
    image1_features = dino_model.encode_image(image1)
    image2_features = dino_model.encode_image(image2)
    return cosine_similarity(image1_features, image2_features)

def evaluate_facesim_similarity(facesim_model, image1, image2):
    face1_features = facesim_model.encode_face(image1)
    face2_features = facesim_model.encode_face(image2)
    return cosine_similarity(face1_features, face2_features)

def evaluate_fgis_similarity(fgis_model, image1, image2):
    image1_features = fgis_model.encode_image(image1)
    image2_features = fgis_model.encode_image(image2)
    return cosine_similarity(image1_features, image2_features)
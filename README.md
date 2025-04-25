Image-Similarity-Metrics
project/
├── models/
│   ├── clip_model.py       # CLIP 模型封裝
│   ├── dino_model.py       # DINO 模型封裝
│   ├── facesim_model.py    # FaceSim 模型封裝
│   ├── fgis_model.py       # FGIS 模型封裝
├── metrics/
│   ├── similarity_metrics.py   # 指標計算函數
├── main.py                # 主程式入口
├── requirements.txt       # 依賴列表
└── README.md              # 項目文檔

A deep learning-based framework for evaluating image similarity using CLIP, DINO, FaceSim, and FGIS models.
Overview
This repository provides a framework for evaluating image similarity based on multiple AI models. It supports text-to-image and image-to-image comparison using CLIP, DINO, FaceSim, and FGIS.
Features

✅ CLIP-based Text-Image Similarity

✅ DINO-based Image-Image Similarity

✅ FaceSim for Face Similarity Evaluation

✅ FGIS for Fine-Grained Image Similarity


Installation
1. Clone the repository
git clone git@github.com:Lica3265/CLIP-DINO-FaceSim-FGIS_Eval.git
cd Image-Similarity-Metrics


2. Create and activate virtual environment
conda create --name image_similarity_env python=3.10 -y
conda activate image_similarity_env


3. Install dependencies
pip install -r requirements.txt



Usage
Run the evaluation script:
python main.py


Ensure that images are stored in the data/ folder before running the script.

Contributing
Feel free to contribute by improving the models or adding new similarity metrics! Create a pull request if you have enhancements.

License
This project is licensed under the MIT License—see the LICENSE file for details

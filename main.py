# main.py
import torch
import os
import numpy as np
from PIL import Image

from model_factory import ModelFactory
from metrics.similarity_metrics import (
    evaluate_clip_text_image_similarity,
    evaluate_clip_image_similarity,
    evaluate_dino_image_similarity,
    evaluate_facesim_similarity,
    evaluate_fgis_similarity
)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 使用工廠模式來初始化模型
    models = {name: ModelFactory.create_model(name, device) for name in ["CLIP", "DINO", "FaceSim", "FGIS"]}

    clip_model = models["CLIP"]
    dino_model = models["DINO"]
    facesim_model = models["FaceSim"]
    fgis_model = models["FGIS"]

    # 設定基準圖片與生成模型目錄
    reference_folder = "data/reference_images/"
    generated_folders = {
        "Me2Meme": "data/Me2Meme/",
        "InstantID": "data/InstantID/",
        "LoRAs": "data/LoRAs/"
    }
    text_input = "A man."
    reference_images = [os.path.join(reference_folder, f) for f in os.listdir(reference_folder) if f.endswith(('.jpg', '.png'))]

    # 儲存所有指標結果
    scores = {model: {"CLIP-T": [], "CLIP-I": [], "DINO": [], "FaceSim": [], "FGIS": []} for model in generated_folders}
    success_counts = {model: 0 for model in generated_folders}  # 計算成功率

    for ref_path in reference_images:
        ref_image = Image.open(ref_path).convert("RGB")
        ref_name = os.path.basename(ref_path)

        for model_name, gen_folder in generated_folders.items():
            gen_image_path = os.path.join(gen_folder, ref_name)
            if os.path.exists(gen_image_path):  # 確保該模型生成了對應的圖片
                success_counts[model_name] += 1

                gen_image = Image.open(gen_image_path).convert("RGB")

                # 計算相似度指標
                clip_t_score = evaluate_clip_text_image_similarity(clip_model, text_input, gen_image)
                clip_i_score = evaluate_clip_image_similarity(clip_model, ref_image, gen_image)
                dino_score = evaluate_dino_image_similarity(dino_model, ref_image, gen_image)
                facesim_score = evaluate_facesim_similarity(facesim_model, ref_image, gen_image)
                fgis_score = evaluate_fgis_similarity(fgis_model, ref_image, gen_image)

                scores[model_name]["CLIP-T"].append(clip_t_score)
                scores[model_name]["CLIP-I"].append(clip_i_score)
                scores[model_name]["DINO"].append(dino_score)
                scores[model_name]["FaceSim"].append(facesim_score)
                scores[model_name]["FGIS"].append(fgis_score)

    # 計算平均值
    avg_scores = {
        model: {
            metric: np.mean(scores[model][metric]) if scores[model][metric] else 0.0
            for metric in scores[model]
        }
        for model in scores
    }

    # 計算成功率
    success_rates = {model: (success_counts[model] / len(reference_images)) * 100 for model in success_counts}

    # 生成 LaTeX 表格
    latex_table = """
    \\begin{table}[h]
        \\centering
        \\begin{tabular}{|c|c|c|c|c|c|c|}
            \\hline
            \\textbf{模型} & \\textbf{CLIP-T (\\%)} & \\textbf{CLIP-I (\\%)} & \\textbf{DINO (\\%)} & \\textbf{FaceSim (\\%)} & \\textbf{FGIS (\\%)} & \\textbf{Success Rate (\\%)} \\\\
            \\hline
    """
    
    for model, metrics in avg_scores.items():
        latex_table += f"{model} & {metrics['CLIP-T'] * 100:.2f} & {metrics['CLIP-I'] * 100:.2f} & {metrics['DINO'] * 100:.2f} & {metrics['FaceSim'] * 100:.2f} & {metrics['FGIS'] * 100:.2f} & {success_rates[model]:.2f} \\\\\n"

    latex_table += """
            \\hline
        \\end{tabular}
        \\caption{各模型的平均相似度與成功率分析}
        \\label{tab:similarity}
    \\end{table}
    """

    # 儲存至 `.tex` 文件
    with open("results.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)
    print("✅ 已完成計算，LaTeX 表格已儲存至 `results.tex`！")
    print("\n⭐ 各模型的平均相似度與成功率分析 ⭐")
    print("| 模型名   | CLIP-T (%) | CLIP-I (%)| DINO (%) | FaceSim (%) | FGIS (%) | Success Rate (%) |")
    print("|---------|-----------|-----------|---------|------------|--------|--------------|")

    for model, metrics in avg_scores.items():
        print(f"| {model} | {metrics['CLIP-T'] * 100:.2f} | {metrics['CLIP-I'] * 100:.2f} |{metrics['DINO'] * 100:.2f} | {metrics['FaceSim'] * 100:.2f} | {metrics['FGIS'] * 100:.2f} | {success_rates[model]:.2f} |")

if __name__ == "__main__":
    main()
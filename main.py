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

    metrics_list = list(next(iter(avg_scores.values())).keys())

    latex_table = "\\begin{table}[h]\n    \\centering\n    \\begin{tabular}{|" + "|".join(["c"] * (len(metrics_list) + 2)) + "|}\n"
    latex_table += "    \\hline\n    \\textbf{模型} & " + " & ".join([f"\\textbf{{{metric} (\\%)}}" for metric in metrics_list]) + " & \\textbf{Success Rate (\\%)} \\\\\n    \\hline\n"

    for model, metrics in avg_scores.items():
        values = " & ".join([f"{metrics[metric] * 100:.2f}" for metric in metrics_list])
        latex_table += f"    {model} & {values} & {success_rates[model]:.2f} \\\\\n"

    latex_table += "    \\hline\n    \\end{tabular}\n    \\caption{各模型的平均相似度與成功率分析}\n    \\label{tab:similarity}\n\\end{table}\n"

    # **輸出 LaTeX 表格**
    with open("results.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)

    header = "| 模型名   | " + " | ".join([f"{metric} (%)" for metric in metrics_list]) + " | Success Rate (%) |"
    separator = "|---------|" + "|".join(["-----------"] * (len(metrics_list) + 1)) + "|"

    print("\n⭐ 各模型的平均相似度與成功率分析 ⭐")
    print(header)
    print(separator)

    for model, metrics in avg_scores.items():
        values = " | ".join([f"{metrics[metric] * 100:.2f}" for metric in metrics_list])
        print(f"| {model} | {values} | {success_rates[model]:.2f} |")


    # for model, metrics in avg_scores.items():
    #     print(f"| {model} | {metrics['CLIP-T'] * 100:.2f} | {metrics['CLIP-I'] * 100:.2f} |{metrics['DINO'] * 100:.2f} | {metrics['FaceSim'] * 100:.2f} | {metrics['FGIS'] * 100:.2f} | {success_rates[model]:.2f} |")

if __name__ == "__main__":
    main()
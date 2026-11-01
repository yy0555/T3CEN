import os
import io
import numpy as np
from PIL import Image
from utils import constants as c


def main():
    import pandas as pd
    import matplotlib.pyplot as plt

    # ========== 参数 ==========
    parquet_path = c.DATA_DIR / "smallnorb" / "smallnorb_test.parquet"   # -> 改成你的 parquet 路径
    row_idx = 100                          # 处理第几行（从0开始）
    side = "image_lt"                    # "image_lt" 或 "image_rt"
    n_top = 1                           # 去掉出现频率最高的 n 个亮度值
    alpha = 0.6                          # 覆盖红色时的透明度 (0-1)

    out_dir = c.OUTPUT_DIR / "v_analysis_outputs"
    os.makedirs(out_dir, exist_ok=True)

    hist_plot_path = os.path.join(out_dir, f"v_hist_original_and_no_top{n_top}.png")
    overlay_path = os.path.join(out_dir, f"overlay_removed_top{n_top}_{side}_row{row_idx}.png")

    # ========== 载入 parquet 并取出图像字节 ==========
    df = pd.read_parquet(parquet_path)

    row = df.loc[row_idx]
    img_bytes = row[side]["bytes"]
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # RGB PIL Image

    # ========== 取 V 通道并统计直方图 ==========
    hsv_np = np.array(img.convert("HSV"))   # shape (H, W, 3)
    v2d = hsv_np[:, :, 2].astype(np.uint8)  # (H, W), 0-255
    v_flat = v2d.ravel()

    # 使用 bincount 统计 0..255 每个亮度值的像素数
    counts = np.bincount(v_flat, minlength=256)  # length 256
    x = np.arange(256)

    # ========== 找出出现次数最多的 n_top 个亮度值并移除它们 ==========
    n_top = int(n_top)
    n_top = max(0, min(n_top, 256))
    # argsort 降序取前 n_top
    top_indices_desc = np.argsort(counts)[::-1]   # 从多到少
    removed_bins = top_indices_desc[:n_top]
    print("将被去掉的亮度值（按频率从高到低）:", removed_bins.tolist())
    print("对应出现次数:", counts[removed_bins].tolist())

    counts_no_top = counts.copy()
    counts_no_top[removed_bins] = 0              # 把这些 bin 的计数设为 0

    # ========== 绘制折线（同时显示原始与去掉后的）并保存 ==========
    plt.figure(figsize=(10, 6))
    plt.plot(x, counts, label="original (all bins)", linewidth=1)
    plt.plot(x, counts_no_top, label=f"removed top {n_top} bins", linewidth=1)
    plt.xlabel("V value (0-255)")
    plt.ylabel("Pixel count")
    plt.title(f"V channel histogram — original and with top {n_top} bins removed")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(hist_plot_path)
    plt.close()
    print(f"已保存直线/柱状对比图: {hist_plot_path}")

    # ========== 生成覆盖图：将被移除亮度像素区域用红色覆盖（alpha 混合） ==========
    img_np = np.array(img).astype(np.float32)  # shape (H, W, 3), float for blending
    h, w = v2d.shape

    # 构造 mask：True 表示该像素的 V 值在 removed_bins 中
    mask = np.isin(v2d, removed_bins)  # shape (H, W), boolean

    # 如果没有任何被去掉的 bin（n_top==0），mask 会全为 False
    if mask.sum() == 0:
        print("注意：没有像素被标记为移除（mask 为空）。将保存原图的复制。")
        overlay_img = Image.fromarray(img_np.astype(np.uint8))
        overlay_img.save(overlay_path)
    else:
        # 红色 [255, 0, 0]
        red = np.array([255.0, 0.0, 0.0], dtype=np.float32)

        # alpha 混合： out = (1-alpha)*orig + alpha*red   (仅在 mask 上)
        blended = img_np.copy()
        # 执行逐像素混合
        blended[mask] = blended[mask] * (1.0 - alpha) + red * alpha

        # 转回 uint8 并保存
        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
        overlay_img = Image.fromarray(blended_uint8)
        overlay_img.save(overlay_path)
        print(f"已保存用红色覆盖被去掉亮度的图片: {overlay_path}")


if __name__ == "__main__":
    main()

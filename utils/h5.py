import os
import h5py
import numpy as np
from PIL import Image
from utils import constants as c


def main():
    # === 参数设置 ===
    path = c.DATA_DIR / "shapes" / "Validation" / "bright255.h5"
    save_dir = c.OUTPUT_DIR / "output_images"   # 保存文件夹
    index =433781    # 想保存的图像编号

    # === 创建保存文件夹 ===
    os.makedirs(save_dir, exist_ok=True)

    # === 打开 h5 文件并读取图像 ===
    with h5py.File(path, "r") as f:
        print("Keys in file:", list(f.keys()))

        images = f["images"]
        print("images shape:", images.shape)
        print("images dtype:", images.dtype)

        # 读取指定编号图像
        img = images[index]
        print(f"读取第 {index} 张图像，shape = {img.shape}, dtype = {img.dtype}")

        # 转为 PIL Image 并保存
        img_pil = Image.fromarray(img)
        save_path = os.path.join(save_dir, f"image_4.png")
        img_pil.save(save_path)

        print(f"✅ 图像已保存到: {save_path}")

        # 如果文件里有标签，也一起显示
        if "labels" in f:
            label = f["labels"][index]
            print("对应标签:", label)


if __name__ == "__main__":
    main()

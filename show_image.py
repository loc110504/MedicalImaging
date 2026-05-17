import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


def normalize_image(img):
    img = np.array(img)

    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        img = (img * 255).astype(np.uint8)

    return img


def create_multiclass_overlay(label_map, class_colors, alpha=0.85, ignore_classes=None):
    label_map = np.array(label_map)
    h, w = label_map.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    if ignore_classes is None:
        ignore_classes = set()
    else:
        ignore_classes = set(ignore_classes)

    for cls_id, color in class_colors.items():
        if cls_id in ignore_classes:
            continue
        mask = label_map == cls_id
        overlay[mask, 0] = color[0]
        overlay[mask, 1] = color[1]
        overlay[mask, 2] = color[2]
        overlay[mask, 3] = alpha

    return overlay


def save_single_image(base_image, save_path, overlay=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(base_image, cmap="gray" if base_image.ndim == 2 else None)

    if overlay is not None:
        ax.imshow(overlay)

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_image_and_scribble_separately(
    h5_path,
    image_key="image",
    scribble_key="scribble",
    save_dir="output_vis",
):
    with h5py.File(h5_path, "r") as f:
        if image_key not in f:
            raise KeyError(f"Không tìm thấy key '{image_key}' trong file")
        if scribble_key not in f:
            raise KeyError(f"Không tìm thấy key '{scribble_key}' trong file")

        image = normalize_image(f[image_key][()])
        scribble = f[scribble_key][()]

    class_colors = {
        1: (1.0, 0.0, 0.0),  # RV
        2: (0.0, 1.0, 0.0),  # Myo
        3: (0.0, 0.4, 1.0),  # LV
    }

    scribble_overlay = create_multiclass_overlay(
        scribble,
        class_colors=class_colors,
        alpha=0.85,
        ignore_classes={0, 4},  # BG, UA
    )

    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(h5_path))[0]

    image_path = os.path.join(save_dir, f"{base_name}_image.png")
    scribble_path = os.path.join(save_dir, f"{base_name}_scribble_overlay.png")

    save_single_image(image, image_path)
    save_single_image(image, scribble_path, overlay=scribble_overlay)

    print("Đã lưu:")
    print(image_path)
    print(scribble_path)


# ===== dùng thử =====
h5_path = "data/MSCMR/MSCMR_training_slices/subject2_DE_slice_8.h5"
save_image_and_scribble_separately(
    h5_path,
    image_key="image",
    scribble_key="scribble",
    save_dir="output_vis",
)
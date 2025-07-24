import os

def count_images_and_labels(base_path):
    sets = ['test', 'train', 'valid']  # <- fixed here
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    for s in sets:
        image_dir = os.path.join(base_path, s, 'images')
        label_dir = os.path.join(base_path, s, 'labels')

        image_count = 0
        label_count = 0

        if os.path.exists(image_dir):
            image_count = sum(1 for f in os.listdir(image_dir) if f.lower().endswith(image_exts))
        else:
            print(f"[!] Missing image folder: {image_dir}")

        if os.path.exists(label_dir):
            label_count = sum(1 for f in os.listdir(label_dir) if f.lower().endswith('.txt'))
        else:
            print(f"[!] Missing label folder: {label_dir}")

        print(f"{s.upper()} SET:")
        print(f"  Images: {image_count}")
        print(f"  Labels: {label_count}")
        print("-" * 30)

# 🔧 Your dataset path
dataset_path = r"C:\Users\Omen\OneDrive\Desktop\project\dataset"
count_images_and_labels(dataset_path)

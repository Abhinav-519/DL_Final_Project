# importing required libraries for data processing
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# We will be setting up data preprocessing functions
def verify_data_pairs(dataset_path: Path, split: str) -> list:
    """We will be verifying and returning valid image-label pairs for given split"""
    print(f"Verifying {split} image-label pairs...")
    valid_pairs = []

    label_files = list(dataset_path.glob(f'labels/{split}/*.txt'))

    for label_path in tqdm(label_files):
        img_path = dataset_path / 'images' / split / f"{label_path.stem}.jpg"

        if img_path.exists():
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                with open(label_path, 'r') as f:
                    labels = f.read().strip().split('\n')
                    if not labels:
                        continue

                valid_pairs.append((img_path, label_path))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    print(f"Found {len(valid_pairs)} valid {split} image-label pairs")
    return valid_pairs

def filter_intersection_scenes(valid_pairs: list) -> list:
    """We will be filtering scenes with either stop signs or multiple vehicles"""
    print("Filtering intersection scenes...")
    intersection_scenes = []

    for img_path, label_path in tqdm(valid_pairs):
        try:
            with open(label_path, 'r') as f:
                labels = [line.strip().split() for line in f]

            has_sign = any(int(float(label[0])) == 8 for label in labels)
            has_traffic_light = any(int(float(label[0])) == 9 for label in labels)
            has_vehicles = any(int(float(label[0])) in [2, 3, 4] for label in labels)

            if (has_sign or has_traffic_light) and has_vehicles:
                intersection_scenes.append((img_path, label_path))

        except Exception as e:
            print(f"Error processing {label_path}: {e}")
            continue

    print(f"Found {len(intersection_scenes)} intersection scenes")
    return intersection_scenes

def process_and_save_data(image_path: Path, label_path: Path, output_dir: Path, img_size: tuple = (640, 640)) -> bool:
    """We will be processing and saving a single image-label pair with resizing"""
    try:
        out_img_dir = output_dir / 'images'
        out_label_dir = output_dir / 'labels'
        out_img_path = out_img_dir / image_path.name
        out_label_path = out_label_dir / label_path.name

        image = cv2.imread(str(image_path))
        resized_image = cv2.resize(image, img_size)

        with open(label_path, 'r') as f:
            labels = f.readlines()

        cv2.imwrite(str(out_img_path), resized_image)

        with open(out_label_path, 'w') as f:
            f.write(''.join(labels))

        return True

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def create_output_structure(output_path: Path) -> None:
    """We will be creating necessary directory structure"""
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            path = output_path / split / subdir
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {path}")

def process_dataset(dataset_path: Path, output_path: Path) -> bool:
    """We will be running main processing pipeline"""
    print("Starting preprocessing pipeline...")
    
    create_output_structure(output_path)

    # We will be processing training data
    print("\nProcessing training data...")
    train_pairs = verify_data_pairs(dataset_path, 'train')
    train_scenes = filter_intersection_scenes(train_pairs)

    print(f"\nProcessing {len(train_scenes)} training scenes...")
    processed_count = 0
    for img_path, label_path in tqdm(train_scenes):
        if process_and_save_data(img_path, label_path, output_path / 'train'):
            processed_count += 1
    print(f"Successfully processed {processed_count}/{len(train_scenes)} training scenes")

    # We will be processing validation data and splitting into val/test
    print("\nProcessing validation data...")
    val_pairs = verify_data_pairs(dataset_path, 'val')
    val_scenes = filter_intersection_scenes(val_pairs)

    val_scenes_split, test_scenes = train_test_split(val_scenes, test_size=0.5, random_state=42)

    print(f"\nProcessing {len(val_scenes_split)} validation scenes...")
    processed_count = 0
    for img_path, label_path in tqdm(val_scenes_split):
        if process_and_save_data(img_path, label_path, output_path / 'val'):
            processed_count += 1
    print(f"Successfully processed {processed_count}/{len(val_scenes_split)} validation scenes")

    print(f"\nProcessing {len(test_scenes)} test scenes...")
    processed_count = 0
    for img_path, label_path in tqdm(test_scenes):
        if process_and_save_data(img_path, label_path, output_path / 'test'):
            processed_count += 1
    print(f"Successfully processed {processed_count}/{len(test_scenes)} test scenes")

    print("\nPreprocessing completed!")
    return True


# We will be visualizing sample data
def visualize_labels(image_path, label_path):
    """We will be visualizing bounding boxes and labels on images"""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    colors = {
        2: (255, 0, 0),    # Car in Red
        8: (0, 255, 0)     # Traffic Sign in Green
    }

    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f]

    for label in labels:
        class_id = int(label[0])
        x_center = float(label[1]) * width
        y_center = float(label[2]) * height
        w = float(label[3]) * width
        h = float(label[4]) * height

        x1 = int(x_center - w/2)
        y1 = int(y_center - h/2)
        x2 = int(x_center + w/2)
        y2 = int(y_center + h/2)

        color = colors.get(class_id, (0, 0, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        class_name = "Car" if class_id == 2 else "Traffic Sign" if class_id == 8 else str(class_id)
        cv2.putText(image, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def draw_labels(image, label_path, class_names):
    """Draw bounding boxes and labels on image"""
    height, width = image.shape[:2]

    # Read labels
    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f]

    # Colors for different classes
    colors = {
        2: (255, 0, 0),    # Car - Red
        3: (0, 255, 0),    # Bus - Green
        4: (0, 0, 255),    # Truck - Blue
        8: (255, 255, 0)   # Traffic sign - Yellow
    }

    # Draw each box
    for label in labels:
        class_id = int(float(label[0]))
        x_center = float(label[1]) * width
        y_center = float(label[2]) * height
        w = float(label[3]) * width
        h = float(label[4]) * height

        # Calculate box coordinates
        x1 = int(x_center - w/2)
        y1 = int(y_center - h/2)
        x2 = int(x_center + w/2)
        y2 = int(y_center + h/2)

        # Get color for class
        color = colors.get(class_id, (128, 128, 128))

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Add label text
        class_name = class_names[class_id]
        label_text = f'{class_name}'
        cv2.putText(image, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    return image

def compare_images(original_dir, processed_dir, num_samples=3):
    """Compare original and processed images side by side with labels"""
    # Class names
    class_names = {
        0: 'person', 1: 'rider', 2: 'car', 3: 'bus',
        4: 'truck', 5: 'bike', 6: 'motor',
        7: 'traffic_light', 8: 'traffic sign', 9: 'train'
    }

    # Get list of processed images
    processed_images = list(Path(processed_dir).glob('**/*.jpg'))

    # Randomly select images
    samples = random.sample(processed_images, min(num_samples, len(processed_images)))

    # Create subplot grid
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 5*num_samples))
    fig.suptitle('Original vs Processed Images Comparison with Labels', fontsize=16)

    for idx, img_path in enumerate(samples):
        # Get corresponding original paths
        original_img_path = Path(original_dir) / 'images' / 'train' / img_path.name
        original_label_path = Path(original_dir) / 'labels' / 'train' / f"{img_path.stem}.txt"
        processed_label_path = Path(processed_dir).parent / 'labels' / f"{img_path.stem}.txt"

        # Read images
        original_img = cv2.imread(str(original_img_path))
        processed_img = cv2.imread(str(img_path))

        # Convert BGR to RGB
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

        # Draw labels
        original_img = draw_labels(original_img.copy(), original_label_path, class_names)
        processed_img = draw_labels(processed_img.copy(), processed_label_path, class_names)

        # Plot images
        axes[idx, 0].imshow(original_img)
        axes[idx, 0].set_title('Original Image with Labels')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(processed_img)
        axes[idx, 1].set_title('Processed Image with Labels')
        axes[idx, 1].axis('off')

        # Add image filename as subtitle
        plt.figtext(0.5, 0.98 - (idx * 0.33), f'Filename: {img_path.name}',
                   ha='center', va='top', fontsize=10)

    plt.tight_layout()
    plt.show()

    # Display image statistics
    for img_path in samples:
        original_path = Path(original_dir) / 'images' / 'train' / img_path.name
        original_img = cv2.imread(str(original_path))
        processed_img = cv2.imread(str(img_path))

        print(f"\nStatistics for {img_path.name}:")
        print(f"Original image shape: {original_img.shape}")
        print(f"Processed image shape: {processed_img.shape}")
        print(f"Original image mean intensity: {original_img.mean():.2f}")
        print(f"Processed image mean intensity: {processed_img.mean():.2f}")
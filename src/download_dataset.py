# download_dataset.py - Direct download ng Cats vs Dogs dataset

import os
import zipfile
import requests
from tqdm import tqdm
import random

print(" CAT VS DOG DATASET DOWNLOADER")
print("=" * 40)

# Setup paths
base_dir = r"C:\Users\Wendel\Projects\Dog-Cat-identifier"
dataset_dir = os.path.join(base_dir, "dogcat-env", "dataset")
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "validation")

# Gumawa ng folders
os.makedirs(os.path.join(train_dir, "cats"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "dogs"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "cats"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "dogs"), exist_ok=True)

print(f" Dataset folder: {dataset_dir}")

# Download URL (Microsoft's Cats vs Dogs dataset)
url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
zip_path = os.path.join(base_dir, "catsdogs.zip")

print("\n⬇  Downloading dataset... (824 MB, mga 5-10 minutes depende sa internet)")
print("   URL:", url)

# Download with progress bar
response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))
block_size = 1024  # 1 KB

with open(zip_path, 'wb') as file, tqdm(
    desc="Downloading",
    total=total_size,
    unit='B',
    unit_scale=True,
    unit_divisor=1024,
) as bar:
    for data in response.iter_content(block_size):
        bar.update(len(data))
        file.write(data)

print(" Download complete!")

# Extract
print("\n Extracting files... (mga 3-5 minutes)")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(base_dir, "temp_extract"))
print(" Extraction complete!")

# Hanapin ang PetImages folder
petimages_path = os.path.join(base_dir, "temp_extract", "PetImages")
if not os.path.exists(petimages_path):
    # Try other possible locations
    for root, dirs, files in os.walk(os.path.join(base_dir, "temp_extract")):
        if "PetImages" in dirs:
            petimages_path = os.path.join(root, "PetImages")
            break

if not os.path.exists(petimages_path):
    print(" Could not find PetImages folder")
    exit()

print(f"📸 Found images at: {petimages_path}")

# Get all cat and dog images
cat_dir = os.path.join(petimages_path, "Cat")
dog_dir = os.path.join(petimages_path, "Dog")

cat_images = [f for f in os.listdir(cat_dir) if f.lower().endswith('.jpg')]
dog_images = [f for f in os.listdir(dog_dir) if f.lower().endswith('.jpg')]

print(f"\n Total images found:")
print(f"    Cats: {len(cat_images)}")
print(f"    Dogs: {len(dog_images)}")

# Shuffle para random
random.shuffle(cat_images)
random.shuffle(dog_images)

# Split: 80% train, 20% validation
cat_split = int(len(cat_images) * 0.8)
dog_split = int(len(dog_images) * 0.8)

# Limit to reasonable numbers (para di masyadong matagal ang training)
max_train = 1000  # 1000 images per class for training
max_val = 200     # 200 images per class for validation

print("\n Organizing images...")

# Copy cat images to train
print(f"   Copying cats to train folder...")
for i, img in enumerate(cat_images[:min(cat_split, max_train)]):
    src = os.path.join(cat_dir, img)
    dst = os.path.join(train_dir, "cats", f"cat_train_{i}.jpg")
    try:
        # Check if file is valid (some images in dataset are corrupted)
        with open(src, 'rb') as f:
            header = f.read(10)
            if header.startswith(b'\xff\xd8'):  # JPEG header
                with open(dst, 'wb') as out:
                    out.write(open(src, 'rb').read())
    except:
        pass

# Copy cat images to validation
for i, img in enumerate(cat_images[cat_split:cat_split+max_val]):
    src = os.path.join(cat_dir, img)
    dst = os.path.join(val_dir, "cats", f"cat_val_{i}.jpg")
    try:
        with open(src, 'rb') as f:
            header = f.read(10)
            if header.startswith(b'\xff\xd8'):
                with open(dst, 'wb') as out:
                    out.write(open(src, 'rb').read())
    except:
        pass

# Copy dog images to train
print(f"   Copying dogs to train folder...")
for i, img in enumerate(dog_images[:min(dog_split, max_train)]):
    src = os.path.join(dog_dir, img)
    dst = os.path.join(train_dir, "dogs", f"dog_train_{i}.jpg")
    try:
        with open(src, 'rb') as f:
            header = f.read(10)
            if header.startswith(b'\xff\xd8'):
                with open(dst, 'wb') as out:
                    out.write(open(src, 'rb').read())
    except:
        pass

# Copy dog images to validation
for i, img in enumerate(dog_images[dog_split:dog_split+max_val]):
    src = os.path.join(dog_dir, img)
    dst = os.path.join(val_dir, "dogs", f"dog_val_{i}.jpg")
    try:
        with open(src, 'rb') as f:
            header = f.read(10)
            if header.startswith(b'\xff\xd8'):
                with open(dst, 'wb') as out:
                    out.write(open(src, 'rb').read())
    except:
        pass

# Count final images
train_cats = len([f for f in os.listdir(os.path.join(train_dir, 'cats')) if f.endswith('.jpg')])
train_dogs = len([f for f in os.listdir(os.path.join(train_dir, 'dogs')) if f.endswith('.jpg')])
val_cats = len([f for f in os.listdir(os.path.join(val_dir, 'cats')) if f.endswith('.jpg')])
val_dogs = len([f for f in os.listdir(os.path.join(val_dir, 'dogs')) if f.endswith('.jpg')])

print("\n" + "=" * 40)
print(" DATASET READY!")
print("=" * 40)
print(f" FINAL COUNT:")
print(f"    Train cats: {train_cats} images")
print(f"    Train dogs: {train_dogs} images")
print(f"    Validation cats: {val_cats} images")
print(f"    Validation dogs: {val_dogs} images")
print(f"\n    Location: {dataset_dir}")
print("=" * 40)

# Clean up temp files
print("\n Cleaning up temporary files...")
os.remove(zip_path)
import shutil
shutil.rmtree(os.path.join(base_dir, "temp_extract"))
print(" Done!")

print("\n Ready na! I-run mo na ang training:")
print("   python src/train_model.py")
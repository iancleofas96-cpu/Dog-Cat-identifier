# balance_dataset.py
import os
import shutil
import random

print("⚖️  BALANCING DATASET")
print("=" * 50)

base_dir = r"C:\Users\Wendel\Projects\Dog-Cat-identifier\dogcat-env\dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

# Current counts
train_cats = [f for f in os.listdir(os.path.join(train_dir, "cats")) if f.endswith('.jpg')]
train_dogs = [f for f in os.listdir(os.path.join(train_dir, "dogs")) if f.endswith('.jpg')]
val_cats = [f for f in os.listdir(os.path.join(val_dir, "cats")) if f.endswith('.jpg')]
val_dogs = [f for f in os.listdir(os.path.join(val_dir, "dogs")) if f.endswith('.jpg')]

print(f"\n BEFORE BALANCING:")
print(f"Train cats: {len(train_cats)}")
print(f"Train dogs: {len(train_dogs)}")
print(f"Val cats: {len(val_cats)}")
print(f"Val dogs: {len(val_dogs)}")

# Target: 200 validation each
target_val = 200

# Kung kulang ang validation cats
if len(val_cats) < target_val:
    needed = target_val - len(val_cats)
    print(f"\n Cats validation: kulang {needed} - kukuha mula sa train")
    
    # Pumili ng random cats mula sa train
    available = train_cats.copy()
    random.shuffle(available)
    
    for i in range(min(needed, len(available) // 5)):  # Huwag ubusin ang train
        src = os.path.join(train_dir, "cats", available[i])
        dst = os.path.join(val_dir, "cats", f"cat_val_moved_{i}.jpg")
        shutil.copy2(src, dst)
        print(f"    Copied {available[i]} to validation")

# Kung kulang ang validation dogs
if len(val_dogs) < target_val:
    needed = target_val - len(val_dogs)
    print(f"\n Dogs validation: kulang {needed} - kukuha mula sa train")
    
    available = train_dogs.copy()
    random.shuffle(available)
    
    for i in range(min(needed, len(available) // 5)):
        src = os.path.join(train_dir, "dogs", available[i])
        dst = os.path.join(val_dir, "dogs", f"dog_val_moved_{i}.jpg")
        shutil.copy2(src, dst)
        print(f"    Copied {available[i]} to validation")

# Final counts
train_cats_final = len(os.listdir(os.path.join(train_dir, "cats")))
train_dogs_final = len(os.listdir(os.path.join(train_dir, "dogs")))
val_cats_final = len(os.listdir(os.path.join(val_dir, "cats")))
val_dogs_final = len(os.listdir(os.path.join(val_dir, "dogs")))

print(f"\n AFTER BALANCING:")
print(f"Train cats: {train_cats_final}")
print(f"Train dogs: {train_dogs_final}")
print(f"Val cats: {val_cats_final}")
print(f"Val dogs: {val_dogs_final}")
print(f"\n Total images: {train_cats_final + train_dogs_final + val_cats_final + val_dogs_final}")
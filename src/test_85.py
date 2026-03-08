# test_85.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os

# I-load ang model
model = tf.keras.models.load_model('cat_dog_classifier_85percent.keras')
print("✅ 85% Model loaded!")

def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return
    
    # Load at preprocess
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)
    confidence = prediction[0][0]
    
    # Display
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    
    if confidence < 0.5:
        confidence_score = (1 - confidence) * 100
        result = "🐱 CAT"
        color = 'green'
    else:
        confidence_score = confidence * 100
        result = "🐶 DOG"
        color = 'blue'
    
    plt.title(f"{result}\n{confidence_score:.1f}% confident", 
              fontsize=18, color=color, fontweight='bold')
    plt.show()
    
    print(f"\n📊 Result: {result}")
    print(f"📈 Confidence: {confidence_score:.1f}%")
    return result

# Test
if __name__ == "__main__":
    img_path = input("Enter image path (or drag & drop): ").strip('"')
    predict_image(img_path)
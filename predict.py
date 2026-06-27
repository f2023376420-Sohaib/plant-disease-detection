import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 1. Model ka path
MODEL_PATH = "deployment/model_v1.h5"

# 2. Data folder ka path jahan aapki bimaariyon ke sub-folders hain
# (Agar data ke andar 'train' ka folder hai, toh yahan "data/train" ya "data/training" kar dein)
DATA_DIR = "data" 

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: '{MODEL_PATH}' nahi mili! Check karen file kahan hai.")
    exit()

# --- SMART WORK: Classes khud hi nikalain ---
try:
    # Folders ke naam nikaal kar alphabetically sort karega (bilkul Keras ki tarah)
    CLASS_NAMES = sorted([f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))])
    print(f"✅ Total {len(CLASS_NAMES)} Classes khud hi detect ho gayi hain!")
    print(f"🌿 Classes List: {CLASS_NAMES}")
except Exception as e:
    print(f"❌ Error: {DATA_DIR} folder read nahi ho raha. Path check karen. Error: {e}")
    exit()

# Model load karen
print("\n🔄 Model load ho raha hai... Thoda intezar karen.")
model = tf.keras.models.load_model(MODEL_PATH)


def predict_leaf_disease(img_path):
    if not os.path.exists(img_path):
        print(f"❌ Error: Image file '{img_path}' nahi mili!")
        return

    print(f"\n🔍 Image check ho rahi hai: {img_path}")

    # Image preprocessing
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalization

    # Model prediction
   # Model prediction
    predictions = model.predict(img_array)
    
    # Raw scores ko asli probabilities (0 se 100%) mein badalna (Softmax function)
    probabilities = tf.nn.softmax(predictions[0]).numpy()
    
    predicted_class_idx = np.argmax(probabilities)
    confidence = np.max(probabilities) * 100

    result = CLASS_NAMES[predicted_class_idx]

    print("\n" + "="*40)
    print(f"🌿 Detected Condition : {result}")
    print(f"📊 Confidence Score  : {confidence:.2f}%")
    print("="*40 + "\n")


# --- TEST IMAGE PATH ---
# 🚨 BAS YAHAN APNI KISI EK ASLI IMAGE KA PATH DEIN JO LAPTOP MEIN HO 🚨
# Example: agar data folder mein koi image pari hai toh uska naam likh dein
sample_image_path = "test.jpg" 

predict_leaf_disease(sample_image_path)
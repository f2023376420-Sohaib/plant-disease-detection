import tensorflow as tf
import mlflow
import mlflow.tensorflow
import os
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
# 1. Dataset aur Image ka size set karein
data_dir = "data"  # Humara data folder
batch_size = 32
img_height = 128
img_width = 128

print("Dataset load ho raha hai...")

# Training Data
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Validation Data (Test karne ke liye)
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Total Bimaariyan (Classes) detect hui hain: {num_classes}")

# 2. MLflow Setup (Sir ki Requirement #5)
mlflow.set_experiment("Plant_Disease_Detection")
mlflow.tensorflow.autolog() # Yeh automatically sab kuch track karega

# 3. Model Version 1 (Baseline Model) Banayein
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 4. Model ko Train karein
print("Training shuru ho rahi hai... (Isme thoda time lag sakta hai)")
epochs = 3  # Abhi hum sirf 3 baar (epochs) data guzarenge taake jaldi check ho jaye

with mlflow.start_run(run_name="Baseline_Model_V1"):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    
    # 5. Model ko save karein taake baad mein API mein use ho sake
    model.save("deployment/model_v1.h5")
    print("\nModel Version 1 successfully train aur save ho gaya hai!")
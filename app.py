import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io

app = FastAPI(title="Plant Disease Detection API", version="2.0")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------
# AUTHENTICATION DATABASE (Temporary in-memory)
# -----------------------------------------
# Real projects mein yahan PostgreSQL ya SQLite use hota hai
users_db = {} 

class UserCredentials(BaseModel):
    email: str
    password: str

@app.post("/register")
def register_user(user: UserCredentials):
    if user.email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered!")
    # Password save karna (Real app mein isay hash kiya jata hai)
    users_db[user.email] = user.password
    return {"message": "Registration successful!"}

@app.post("/login")
def login_user(user: UserCredentials):
    if user.email not in users_db or users_db[user.email] != user.password:
        raise HTTPException(status_code=401, detail="Invalid email or password!")
    return {"message": "Login successful!", "user": user.email}


# -----------------------------------------
# ML MODEL SETUP 
# -----------------------------------------
MODEL_PATH = "deployment/model_v1.h5"
DATA_DIR = "data"

print("🔄 Model load ho raha hai...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model successfully loaded!")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

try:
    if os.path.exists(DATA_DIR):
        CLASS_NAMES = sorted([f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))])
    else:
        CLASS_NAMES = []
except Exception as e:
    CLASS_NAMES = []

# -----------------------------------------
# PREDICTION ENDPOINT
# -----------------------------------------
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not model:
        return JSONResponse(status_code=500, content={"message": "Model not loaded properly on server."})
    
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"message": "Please upload a valid image file!"})

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((128, 128))
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        
        predicted_class_idx = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities) * 100)

        result_class = CLASS_NAMES[predicted_class_idx] if (CLASS_NAMES and predicted_class_idx < len(CLASS_NAMES)) else "Unknown"

        return {
            "prediction": result_class,
            "confidence": f"{confidence:.2f}%"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Prediction Error: {str(e)}"})
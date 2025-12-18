from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from backend.inference import load_model, predict_image
from backend.database import init_db, save_prediction
from backend.recommender import get_recommendation
from backend.schemas import PredictionResponse

app = FastAPI(title="Plant Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
def startup_event():
    global model
    init_db()
    model = load_model()
    print("✅ Model loaded successfully")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    prediction = predict_image(model, image)

    # Recommendation logic
    if "healthy" in prediction["disease"].lower():
        recommendation = "Plant is healthy. No treatment required."
    else:
        recommendation = get_recommendation(
            prediction["crop"],
            prediction["disease"]
        )

    # Save to DB
    save_prediction(
        crop=prediction["crop"],
        disease=prediction["disease"],
        confidence=prediction["confidence"]
    )

    return {
        "crop": prediction["crop"],
        "disease": prediction["disease"],
        "confidence": prediction["confidence"],
        "recommendation": recommendation
    }

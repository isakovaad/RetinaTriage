"""RetinaTriage backend API.

Run from inside the backend/ folder:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000/docs and try /predict with a fundus image.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import predict_from_bytes

app = FastAPI(title="RetinaTriage API", version="0.1.0")

# Let the Next.js dev server (added later) call this during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    image_bytes = await file.read()
    try:
        # The model loads lazily on the first request, then stays in memory.
        return predict_from_bytes(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

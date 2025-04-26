from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os

app = FastAPI()

# تنظیم CORS برای اجازه دادن به درخواست‌ها از اپ Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # در محیط تولید، دامنه اپ رو مشخص کن
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# لود کردن مدل چندزبانه
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.post("/embed")
async def get_embedding(text: dict):
    try:
        # بررسی اینکه متن ارسال شده
        if "text" not in text:
            raise HTTPException(status_code=400, detail="Text field is required")
        
        input_text = text["text"]
        if not input_text or not isinstance(input_text, str):
            raise HTTPException(status_code=400, detail="Text must be a non-empty string")

        # تولید Embedding
        embedding = model.encode(input_text, convert_to_numpy=True).tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

@app.get("/")
async def root():
    return {"message": "NLP Server is running"}

# برای Render، پورت رو از متغیر محیطی می‌خونیم
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Render پورت رو از متغیر PORT می‌ده
    uvicorn.run(app, host="0.0.0.0", port=port)

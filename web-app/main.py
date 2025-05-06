# main.py

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from image_generator import (
    load_custom_pipeline,
    load_base_pipeline,
    generate_image,
    pil_image_to_base64_str
)

# Load pipelines once
pipe_custom = load_custom_pipeline(
    repo_id="xcheng20/stable-diffusion-painting-style-v1",
    subfolder="fine-tuned-model"
)
pipe_base = load_base_pipeline("CompVis/stable-diffusion-v1-4")

app = FastAPI()

# CORS (if your frontend ever runs from a different origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve your frontend/ folder at the root URL
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


class PromptData(BaseModel):
    prompt: str


@app.post("/generate-images")
async def generate_images_endpoint(data: PromptData):
    if pipe_custom is None or pipe_base is None:
        raise HTTPException(status_code=500, detail="Pipelines are not loaded.")
    try:
        images_base64 = []
        # two from custom
        for i in range(2):
            img = generate_image(pipe_custom, data.prompt, seed=42 + i)
            images_base64.append(pil_image_to_base64_str(img))
        # one from base
        img = generate_image(pipe_base, data.prompt, seed=100)
        images_base64.append(pil_image_to_base64_str(img))

        return JSONResponse(content={"images": images_base64})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating images: {str(e)}")

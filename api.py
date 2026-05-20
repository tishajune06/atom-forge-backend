from fastapi import FastAPI
from pydantic import BaseModel
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from generate import load_vae, load_gnn
from utils.optimization import optimize_latent_space


app = FastAPI()

# Allow frontend access from localhost and Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


device = torch.device("cpu")

vae_model, idx_to_token = load_vae(device)
gnn_model, property_names = load_gnn(device)


class Request(BaseModel):
    property: str
    target: float
    samples: int = 1000


@app.post("/generate")
def generate(req: Request):
    results = optimize_latent_space(
        vae_model=vae_model,
        gnn_model=gnn_model,
        idx_to_token=idx_to_token,
        target_property=req.target,
        selected_property=req.property,
        num_samples=req.samples,
        device=device,
    )

    return {"results": results}


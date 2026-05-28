from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import os
import torch

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED

from generate import load_vae, load_gnn
from utils.optimization import optimize_latent_space

import sas  # keep only if installed correctly

# ---------------- APP SETUP ----------------
app = FastAPI()

# ---------------- ROOT ENDPOINT ----------------
@app.get("/")
def home():
    return {
        "message": "Drug Discovery API is running"
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELS ----------------
device = torch.device("cpu")

vae_model, idx_to_token = load_vae(device)
gnn_model, property_names = load_gnn(device)

# ---------------- INPUT SCHEMAS ----------------
class MoleculeInput(BaseModel):
    smiles: str


class GenerateRequest(BaseModel):
    property: str
    target: float
    samples: int = 1000


# ---------------- PREDICT ENDPOINT ----------------
@app.post("/predict")
def predict(data: MoleculeInput):

    smiles = data.smiles.strip()
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {
            "valid": False,
            "smiles": smiles
        }

    return {
        "valid": True,
        "smiles": smiles,

        # RDKit descriptors
        "logp": float(Crippen.MolLogP(mol)),
        "qed": float(QED.qed(mol)),
        "sas": float(sas.calculateScore(mol)),

        "mw": float(Descriptors.MolWt(mol)),
        "hbd": int(Descriptors.NumHDonors(mol)),
        "hba": int(Descriptors.NumHAcceptors(mol)),
        "tpsa": float(Descriptors.TPSA(mol))
    }


# ---------------- GENERATE ENDPOINT ----------------
@app.post("/generate")
def generate(req: GenerateRequest):

    results = optimize_latent_space(
        vae_model=vae_model,
        gnn_model=gnn_model,
        idx_to_token=idx_to_token,
        target_property=float(req.target),
        selected_property=req.property,
        num_samples=req.samples,
        device=device,
    )

    return {"results": results}
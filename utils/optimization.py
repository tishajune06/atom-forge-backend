#%%writefile utils/optimization.py
"""
Latent-space optimization utilities.

This module:
1. Decodes random latent vectors using the trained VAE.
2. Validates generated SMILES strings.
3. Uses the multi-property GNN to predict properties.
4. Ranks molecules by closeness to a user-selected target.
"""
# Add this import at the top of utils/optimization.py
import base64
import requests
from utils.chemistry import draw_molecule
import torch

SESSION = requests.Session()

from utils.chemistry import (
    is_valid_smiles,
    canonicalize_smiles,
)

from utils.graph_utils import PROPERTY_NAMES, smiles_to_graph
import selfies as sf
# Add these imports at the TOP of utils/optimization.py

from rdkit import Chem

from rdkit.Chem import rdMolDescriptors
# ------------------------------------------------------
# Decode Token IDs -> SMILES
# ------------------------------------------------------
def decode_tokens(tokens, idx_to_token):
    chars = []

    for idx in tokens.tolist():
        if idx == 0:
            continue

        if 0 <= idx < len(idx_to_token):
            token = idx_to_token[idx]

            if token is None:
                continue

            if token == "<PAD>":
                continue

            chars.append(token)

    return "".join(chars)


# ------------------------------------------------------
# Predict Selected Property
# ------------------------------------------------------
def predict_property(
    smiles,
    gnn_model,
    selected_property="QED",
    device="cpu",
):
    # Validate property name
    if selected_property not in PROPERTY_NAMES:
        raise ValueError(
            f"Unsupported property: {selected_property}"
        )

    # Convert molecule to graph
    graph = smiles_to_graph(smiles)

    if graph is None:
        return None

    graph = graph.to(device)

    # Single-graph batch vector
    batch = torch.zeros(
        graph.x.size(0),
        dtype=torch.long,
        device=device,
    )

    # Determine output index
    property_index = PROPERTY_NAMES.index(
        selected_property
    )

    # Predict all properties
    with torch.no_grad():
        pred = gnn_model(
            graph.x,
            graph.edge_index,
            batch,
        )

    # Select the requested property
    value = pred[0, property_index].item()

    return float(value)

import requests

import requests
import selfies as sf
from urllib.parse import quote
NAME_CACHE = {}


# Add this helper function to utils/optimization.py

def smiles_to_base64_image(smiles, filename="temp_molecule.png"):
    """
    Generate a PNG image from SMILES and return it as a Base64 string.
    This allows the frontend to display the molecule directly.
    """
    # Draw molecule using your existing RDKit function
    success = draw_molecule(smiles, filename)

    if not success:
        return None

    # Read image and encode as Base64
    with open(filename, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return encoded

# ------------------------------------------------------
# Add this to utils/optimization.py
# ------------------------------------------------------

# Optional friendly names for common molecules.
# If a molecule is not listed here, the canonical SMILES
# itself will be used as the identifier.
COMMON_NAMES = {
    "CCO": "Ethanol",
    "c1ccccc1": "Benzene",
    "CC(=O)O": "Acetic acid",
    "CC(C)O": "Isopropanol",
    "CCN": "Ethylamine",
    "CCOCC": "Diethyl ether",
}


# Add these imports at the top of utils/optimization.py
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


# ------------------------------------------------------
# Friendly names for well-known molecules
# Key = canonical SMILES
# Value = rich naming information
# ------------------------------------------------------
KNOWN_MOLECULES = {
    "O=O": {
        "systematic_name": "Dioxygen",
        "iupac_name": "Dioxygen",
        "common_name": "Oxygen gas",
    },
    "N#N": {
        "systematic_name": "Dinitrogen",
        "iupac_name": "Dinitrogen",
        "common_name": "Nitrogen gas",
    },
    "CCO": {
        "systematic_name": "Ethanol",
        "iupac_name": "Ethanol",
        "common_name": "Ethyl alcohol",
    },
    "c1ccccc1": {
        "systematic_name": "Benzene",
        "iupac_name": "Benzene",
        "common_name": "Benzene",
    },
    "CC(=O)O": {
        "systematic_name": "Ethanoic acid",
        "iupac_name": "Ethanoic acid",
        "common_name": "Acetic acid",
    },
    "CO": {
        "systematic_name": "Methanol",
        "iupac_name": "Methanol",
        "common_name": "Methyl alcohol",
    },
    "C": {
        "systematic_name": "Methane",
        "iupac_name": "Methane",
        "common_name": "Natural gas",
    },
}

MOLECULE_NAME_CACHE = {}
def molecule_name(smiles):
    """
    Get real chemical names from PubChem.

    Returns:
        {
            "name": ...,
            "systematic_name": ...,
            "iupac_name": ...,
            "common_name": ...,
            "formula": ...
        }

    Strategy:
    1. Convert to canonical SMILES.
    2. Query PubChem for IUPAC and common names.
    3. Use molecular formula as fallback if PubChem fails.

    This works perfectly with SELFIES because the SELFIES are decoded
    into valid SMILES before this function is called.
    """

    # -----------------------------
    # Validate molecule
    # -----------------------------
    
    if smiles in MOLECULE_NAME_CACHE:
          return MOLECULE_NAME_CACHE[smiles]

    # Default empty result
    empty_result = {
        "name": None,
        "systematic_name": None,
        "iupac_name": None,
        "common_name": None,
        "formula": None,
    }
    if not smiles:
        MOLECULE_NAME_CACHE[smiles] = empty_result
        return empty_result

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {
            "name": None,
            "systematic_name": None,
            "iupac_name": None,
            "common_name": None,
            "formula": None,
        }

    # Canonical SMILES
    canonical = Chem.MolToSmiles(mol, canonical=True)

    # Molecular formula (always available)
    formula = rdMolDescriptors.CalcMolFormula(mol)

    # -----------------------------
    # PubChem Query
    # -----------------------------
    try:
        # URL-encode the SMILES safely
        from urllib.parse import quote
        encoded = quote(canonical, safe="")

        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            f"compound/smiles/{encoded}/property/"
            "IUPACName,Title,MolecularFormula/JSON"
        )

        # Short timeout prevents hanging
        response = SESSION.get(url, timeout=3)

        if response.status_code == 200:
            data = response.json()

            props = data["PropertyTable"]["Properties"][0]

            iupac_name = props.get("IUPACName")
            title = props.get("Title")
            formula = props.get("MolecularFormula", formula)

            # Prefer common title when available, otherwise IUPAC
            display_name = title or iupac_name or formula

            return {
                "name": display_name,
                "systematic_name": iupac_name or formula,
                "iupac_name": iupac_name or formula,
                "common_name": title or iupac_name or formula,
                "formula": formula,
            }

    except Exception:
        # Network issue, timeout, or molecule not found
        pass

    # -----------------------------
    # Fallback if PubChem fails
    # -----------------------------
    return {
        "name": formula,
        "systematic_name": formula,
        "iupac_name": formula,
        "common_name": formula,
        "formula": formula,
    }


def smiles_to_name(smiles):
    """
    Return the IUPAC/common name for a SMILES string using PubChem.

    Features:
    1. Looks up only when explicitly called.
    2. Caches previous results in memory.
    3. Handles all network/API errors safely.
    4. Returns None if no name is found.
    """
    # Return cached result instantly
    if smiles in NAME_CACHE:
        return NAME_CACHE[smiles]

    try:
        encoded_smiles = quote(smiles, safe="")

        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            f"compound/smiles/{encoded_smiles}/property/IUPACName/JSON"
        )

        response = SESSION.get(url, timeout=3)

        if response.status_code != 200:
            NAME_CACHE[smiles] = None
            return None

        data = response.json()

        name = data["PropertyTable"]["Properties"][0]["IUPACName"]

        # Save to cache
        NAME_CACHE[smiles] = name

        return name

    except Exception:
        NAME_CACHE[smiles] = None
        return None





# ------------------------------------------------------
# Random Latent Search
# ------------------------------------------------------
def optimize_latent_space(
    vae_model,
    gnn_model,
    idx_to_token,
    target_property,
    selected_property="QED",
    num_samples=1000,
    device="cpu",
):
    """
    Randomly sample the VAE latent space and return
    the molecules whose predicted property is closest
    to the requested target value.
    """
    
    vae_model.eval()
    gnn_model.eval()
    
    from utils.chemistry import mol_to_base64
    
    

    results = []
    seen = set()

    with torch.no_grad():
        for _ in range(num_samples):
            # Random latent vector
            z = torch.randn(
                1,
                vae_model.latent_dim,
                device=device,
            )

            # Decode into token logits
            logits = vae_model.decode(z)

            # Greedy decoding
            tokens = torch.argmax(
                logits,
                dim=-1,
            )[0]

            # Convert tokens to SMILES
            selfies = decode_tokens(
                tokens,
                idx_to_token
            )
            
            smiles = sf.decoder(selfies)
            
            if not smiles:
                continue

            # Validate molecule
            if not is_valid_smiles(smiles):
                continue

            # Canonicalize
            smiles = canonicalize_smiles(smiles)

            if smiles is None:
                continue

            # Remove duplicates
            if smiles in seen:
                continue

            ##name_info = molecule_name(smiles)
            seen.add(smiles)

            # Predict selected property
            predicted = predict_property(
                smiles=smiles,
                gnn_model=gnn_model,
                selected_property=selected_property,
                device=device,
            )

            if predicted is None:
                continue

            # Distance to target
            error = abs(
                predicted - target_property
            )
            accuracy = max(0, 1 - error)
            
            

            # Store result
            results.append({
                 "smiles": smiles,
                 "selected_property": selected_property,
                 "predicted_property": predicted,
                 "target_property": target_property,
                 "error": error,
                 "accuracy": accuracy * 100,
})
            


    import heapq
    top_results = heapq.nsmallest(10, results, key=lambda x: x["error"])

    for result in top_results:
        name_info = molecule_name(result["smiles"])
        result["name"] = name_info["name"]
        result["systematic_name"] = name_info["systematic_name"]
        result["iupac_name"] = name_info["iupac_name"]
        result["common_name"] = name_info["common_name"]
        result["formula"] = name_info["formula"]
        result["image_base64"] = smiles_to_base64_image(
        result["smiles"]
    )
        

    return top_results

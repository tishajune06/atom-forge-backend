from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED
import sas

def compute_rdkit(smiles: str):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return {
        "valid": True,

        "logp": float(Crippen.MolLogP(mol)),
        "qed": float(QED.qed(mol)),
        "sas": float(sas.calculateScore(mol)),

        "mw": float(Descriptors.MolWt(mol)),
        "hbd": int(Descriptors.NumHDonors(mol)),
        "hba": int(Descriptors.NumHAcceptors(mol)),
        "tpsa": float(Descriptors.TPSA(mol))
    }
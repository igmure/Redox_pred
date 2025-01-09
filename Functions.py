import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, PandasTools, AllChem, MACCSkeys, AtomPairs, rdFingerprintGenerator
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdmolops import PatternFingerprint
from mordred import Calculator, descriptors
import os
import tqdm


import pandas as pd
import numpy as np
import warnings

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, PandasTools, MACCSkeys, AtomPairs, rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.Chem.rdmolops import PatternFingerprint
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprintAsBitVect
import mordred
pd.set_option('display.max_rows', None)

warnings.filterwarnings("ignore")

def generate_0D_descriptors(mol):
    return {
        'MolWt': Descriptors.MolWt(mol),
        'NumAtoms': mol.GetNumAtoms(),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol)
    }

# Function to generate 1D descriptors
def generate_1D_descriptors(mol):
    return {
        'NumRings': Descriptors.RingCount(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol)
    }

# Function to generate 2D descriptors
def generate_2D_descriptors(mol):
    descriptors = {
        'TPSA': Descriptors.TPSA(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHBD': rdMolDescriptors.CalcNumHBD(mol),
        'NumHBA': rdMolDescriptors.CalcNumHBA(mol),
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol)
    }
    return descriptors


def generate_descriptors(mol, row, number, red):
    if mol is None:
        return None

    # Calculate 0D, 1D, and 2D descriptors using RDKit
    rdkit_descriptors = {
        'System_number': number,
        'dG_red': red,
        'MolWt': Descriptors.MolWt(mol),
        'NumAtoms': mol.GetNumAtoms(),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
        'NumRings': Descriptors.RingCount(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'TPSA': Descriptors.TPSA(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHBD': rdMolDescriptors.CalcNumHBD(mol),
        'NumHBA': rdMolDescriptors.CalcNumHBA(mol),
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol)
    }

    # Combine Mordred and RDKit descriptors
    descriptors = {**rdkit_descriptors}

    return descriptors
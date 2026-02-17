#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 23:22:46 2026

@author: aaron.sweeney
"""

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem


@pytest.fixture(scope="session", autouse=True)
def _silence_rdkit():
    RDLogger.DisableLog("rdApp.*")


@pytest.fixture
def mol3d():
    """Factory fixture: build a small 3D molecule deterministically."""
    def _make(smiles: str, seed: int = 0xC0FFEE):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        res = AllChem.EmbedMolecule(mol, randomSeed=seed)
        assert res == 0
        AllChem.UFFOptimizeMolecule(mol, maxIters=50)
        return mol
    return _make


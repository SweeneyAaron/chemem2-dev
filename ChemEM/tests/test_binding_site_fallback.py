"""Regression tests for the SES ray-trace fallback binding site.

When a user-supplied ``centroid =`` lands inside no alpha-shape pocket cluster,
``BindingSite.get_centroid_binding_sites`` builds the site with
``ses_ray_trace_binding_site`` instead of the primary cluster path. That fallback used to
write its lining molecule to ``site_data.lining_mol`` -- a name ``BindingSiteModel`` does
not declare. The dataclass is not slotted, so the assignment silently created a stray
attribute and left ``rdkit_lining_mol`` at its ``None`` default; every ``--dock`` run whose
centroid took the fallback then died in ``PreCompDataProtein`` with a bare
``'NoneType' object has no attribute 'GetConformer'``.

Two things are pinned here:

  * no assignment in ``ses_ray_trace_binding_site`` may target an undeclared
    ``BindingSiteModel`` attribute -- this is the exact typo class, and on a non-slotted
    dataclass Python will never catch it at runtime;
  * ``PreCompDataProtein`` must name the offending site rather than dereference ``None``,
    so the next instance of this is diagnosable from the log alone.

Run with:  pytest ChemEM/tests/test_binding_site_fallback.py     (env: chemem2-run)
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
import types

import numpy as np
import pytest

from ChemEM.data.binding_site_model import BindingSiteModel
from ChemEM.tools import binding_site as bs_tools


def _field_names() -> set[str]:
    return {f.name for f in dataclasses.fields(BindingSiteModel)}


# --------------------------------------------------------------------------------------
# the field the fallback has to populate
# --------------------------------------------------------------------------------------
def test_binding_site_model_declares_rdkit_lining_mol_and_not_lining_mol():
    names = _field_names()
    assert "rdkit_lining_mol" in names
    assert "rdkit_mol" in names
    # If someone ever adds `lining_mol` as a real field, the two names diverge again and
    # consumers like PreCompDataProtein keep reading the one nobody populates.
    assert "lining_mol" not in names


def test_ses_fallback_only_assigns_declared_binding_site_fields():
    """Every `site_data.<attr> = ...` in the fallback must hit a real dataclass field."""
    src = textwrap.dedent(inspect.getsource(bs_tools.ses_ray_trace_binding_site))
    tree = ast.parse(src)

    assigned = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "site_data"
            ):
                assigned.add(target.attr)

    # sanity: the walk actually found the assignments it is meant to police
    assert "rdkit_lining_mol" in assigned, assigned

    unknown = assigned - _field_names()
    assert not unknown, (
        f"ses_ray_trace_binding_site assigns undeclared BindingSiteModel "
        f"attribute(s) {sorted(unknown)}; BindingSiteModel is not slotted, so these are "
        f"silently dropped and the declared field keeps its default."
    )


# --------------------------------------------------------------------------------------
# the consumer must name the site instead of dereferencing None
# --------------------------------------------------------------------------------------
def _minimal_options() -> types.SimpleNamespace:
    """Just enough of the argparse Namespace to reach the lining-mol guard."""
    return types.SimpleNamespace(
        ncpu=1,
        n_global_search=1,
        n_local_search=1,
        repulsion_cap_0=1.0,
        repulsion_cap_1=1.0,
        repulsion_cap_nm=1.0,
        repulsion_cap_polish=1.0,
        return_n=1,
        max_iterations=1,
        inner_map_score=0,
        outer_map_score=0,
    )


def test_precomp_protein_names_the_site_when_lining_mol_missing():
    from ChemEM.tools.precomputed_data import PreCompDataProtein

    site = BindingSiteModel.from_dict(
        {
            "key": 7,
            "source": "Voxel Mask Analysis",
            "lining_residues": [],
            "distance_map": np.zeros((2, 2, 2)),
        }
    )
    assert site.rdkit_lining_mol is None

    system = types.SimpleNamespace(ncpu=1, options=_minimal_options())

    with pytest.raises(ValueError) as exc:
        PreCompDataProtein(site, system)

    msg = str(exc.value)
    assert "rdkit_lining_mol" in msg
    assert "7" in msg
    assert "Voxel Mask Analysis" in msg

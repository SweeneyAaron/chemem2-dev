"""Regression tests for `--manual-site`, the centroid-defined binding site.

The manual site exists because the automatic path can only hand docking a volume that
alpha-shape clustering and density segmentation agree on. When they disagree the pocket
gets split, a site box ends up clipping part of the ligand, and a decoy site can outscore
the real one. `--manual-site` lets the user state the volume outright.

Three things are pinned here, all of which fail *silently* rather than loudly:

  * **the orientation contract** -- `distance_map` is (z,y,x) while `origin`/`apix` are
    (x,y,z), and `box_size` must be (nz,ny,nx) because `handle_centroids` subscripts it
    reversed. Get any of these wrong and the ACO searches a mirrored or transposed
    volume with no error anywhere;
  * **requested extent == searched extent** -- the ACO takes its translation points from
    `distance_map > probe_radius`, so encoding a real boundary EDT would erode the probe
    radius off every face and a requested 10 Å box would quietly become 7.2 Å;
  * **box wins over radius**, and a site that would be built empty returns None instead
    of a half-populated model.

Run with:  pytest ChemEM/tests/test_manual_binding_site.py     (env: chemem2-run)
"""

from __future__ import annotations

import numpy as np
import pytest

from ChemEM.data.binding_site_model import BindingSiteModel
from ChemEM.tools.binding_site import manual_binding_site, resolve_manual_extent

PROBE = 1.4


# --------------------------------------------------------------------------------------
# minimal ParmEd-shaped fakes: enough for write_residues_to_pdb and the KD-tree selection
# --------------------------------------------------------------------------------------
class _Pos:
    def __init__(self, xyz):
        self.x, self.y, self.z = (float(v) for v in xyz)


class _Res:
    def __init__(self, name, number):
        self.name = name
        self.number = number
        self.idx = number - 1
        self.chain = "A"
        self.atoms = []


class _Atom:
    def __init__(self, idx, name, element_name, atomic_number, residue):
        self.idx = idx
        self.name = name
        self.element_name = element_name
        self.element = atomic_number
        self.atomic_number = atomic_number
        self.bond_partners = []          # no bonds -> no INTRA_RESIDUE_BOND_DATA lookups
        self.residue = residue
        residue.atoms.append(self)


class _Structure:
    def __init__(self, positions):
        self.positions = [_Pos(p) for p in positions]


def _fake_protein(coords):
    """One carbon per coordinate, two atoms per residue."""
    atoms, residues = [], []
    for i, _ in enumerate(coords):
        if i % 2 == 0:
            residues.append(_Res("ALA", len(residues) + 1))
        atoms.append(_Atom(i, "CA" if i % 2 == 0 else "CB", "C", 6, residues[-1]))
    positions = np.asarray(coords, dtype=float)
    radii = np.full(len(coords), 1.7)
    return atoms, positions, radii, _Structure(positions)


def _shell(centre, radius, n=60, seed=0):
    """Roughly uniform points on a sphere -- a wall of protein around the centroid."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.asarray(centre, dtype=float)[None, :] + v * radius


def _build(coords, centroid, tmp_path, **kw):
    atoms, positions, radii, struct = _fake_protein(coords)
    return manual_binding_site(
        centroid=np.asarray(centroid, dtype=float),
        centroid_key=kw.pop("key", 0),
        atoms=atoms,
        positions=positions,
        atom_radii=radii,
        protein_openff_structure=struct,
        system_output_dir=str(tmp_path),
        grid_spacing=kw.pop("grid_spacing", 0.5),
        probe_radius=PROBE,
        write_files=False,
        **kw,
    )


# --------------------------------------------------------------------------------------
# extent resolution
# --------------------------------------------------------------------------------------
def test_box_wins_over_radius_and_says_so():
    box, radius, label = resolve_manual_extent(radius=9.0, box=[20.0])
    assert radius is None
    assert np.allclose(box, [20.0, 20.0, 20.0])       # scalar -> cube
    assert "ignored" in label and "9.0" in label       # the dropped radius is surfaced


def test_radius_only_and_three_value_box():
    box, radius, _ = resolve_manual_extent(radius=7.5, box=None)
    assert box is None and radius == 7.5

    box, radius, _ = resolve_manual_extent(radius=None, box=[10.0, 12.0, 14.0])
    assert radius is None and np.allclose(box, [10.0, 12.0, 14.0])


@pytest.mark.parametrize("radius,box", [(None, None), (-1.0, None), (None, [0.0]),
                                        (None, [1.0, 2.0])])
def test_bad_extents_raise(radius, box):
    with pytest.raises(ValueError):
        resolve_manual_extent(radius=radius, box=box)


# --------------------------------------------------------------------------------------
# the orientation contract -- unpinned by any other test in the repo
# --------------------------------------------------------------------------------------
def test_box_size_is_zyx_while_origin_and_apix_are_xyz(tmp_path):
    centroid = np.array([10.0, 20.0, 30.0])
    # deliberately anisotropic, so a transposed axis cannot pass by coincidence
    site = _build(_shell(centroid, 30.0), centroid, tmp_path,
                  box=[6.0, 10.0, 14.0], grid_spacing=0.5)

    nz, ny, nx = site.box_size
    assert (nx, ny, nz) == site.distance_map.shape[::-1]
    assert site.distance_map.shape == (nz, ny, nx)
    # x is the shortest request, z the longest -> nx < ny < nz
    assert nx < ny < nz

    # origin is xyz and brackets the centroid on every axis
    origin = np.asarray(site.origin, dtype=float)
    assert np.all(origin < centroid)
    assert np.all(np.asarray(site.max_coords) > centroid)
    assert np.allclose(site.binding_site_centroid, centroid)


def test_origin_compares_against_a_plain_list_centroid(tmp_path):
    """AlphaMask.handle_centroids does `centroid > site.origin` where the centroid is
    the raw config list. `list > tuple` is a TypeError that kills the whole protocol;
    `list > ndarray` broadcasts. Reproduce that comparison exactly."""
    centroid = [10.0, 20.0, 30.0]                      # a list, as the config supplies
    site = _build(_shell(np.array(centroid), 30.0), centroid, tmp_path, box=[8.0])

    bs = site.box_size
    min_coords = site.origin
    max_coords = site.origin + (site.apix * np.array([bs[2], bs[1], bs[0]]))

    assert np.all(centroid > min_coords)
    assert np.all(centroid < max_coords)


def test_translation_points_round_trip_into_the_requested_box(tmp_path):
    """origin/apix must reconstruct real coordinates from the (z,y,x) voxel indices."""
    from ChemEM.tools.precomputed_data import (covert_idx_to_coords,
                                               get_valid_points_and_adjacency)

    centroid = np.array([10.0, 20.0, 30.0])
    site = _build(_shell(centroid, 30.0), centroid, tmp_path,
                  box=[8.0, 12.0, 16.0], grid_spacing=0.5)

    points, _ = get_valid_points_and_adjacency(site.distance_map > PROBE,
                                               connectivity=26.0)
    coords = covert_idx_to_coords(np.array(points), site.origin, site.apix)

    assert len(coords) > 0
    assert np.all(coords >= np.asarray(site.min_coords) - 1e-6)
    assert np.all(coords <= np.asarray(site.max_coords) + 1e-6)
    # and they surround the centroid rather than sitting off to one side
    assert np.allclose(coords.mean(axis=0), centroid, atol=0.5)


def test_requested_extent_is_the_searched_extent(tmp_path):
    """No probe erosion: a 12 A box must not become a 12 - 2*1.4 A search region."""
    from ChemEM.tools.precomputed_data import (covert_idx_to_coords,
                                               get_valid_points_and_adjacency)

    centroid = np.array([0.0, 0.0, 0.0])
    box = 12.0
    site = _build(_shell(centroid, 40.0), centroid, tmp_path,
                  box=[box], grid_spacing=0.5)

    points, _ = get_valid_points_and_adjacency(site.distance_map > PROBE,
                                               connectivity=26.0)
    coords = covert_idx_to_coords(np.array(points), site.origin, site.apix)
    span = coords.max(axis=0) - coords.min(axis=0)

    assert np.all(span > box - 1.0), f"span {span} eroded well below the requested {box}"
    eroded = box - 2 * PROBE
    assert np.all(span > eroded + 1.0), (
        f"span {span} looks probe-eroded (a boundary EDT would give ~{eroded})")


# --------------------------------------------------------------------------------------
# dockability contract
# --------------------------------------------------------------------------------------
def test_manual_site_populates_everything_docking_reads(tmp_path):
    # Off-lattice centroid on purpose: with a round one (e.g. 5,5,5 and a 10 A box) the
    # origin lands exactly on (0,0,0) and cannot be told apart from the unset default.
    centroid = np.array([5.0, 6.5, 7.25])
    site = _build(_shell(centroid, 9.0, n=120), centroid, tmp_path,
                  box=[10.0], grid_spacing=0.5, lining_distance=4.0, key=3)

    assert isinstance(site, BindingSiteModel)
    assert site.key == 3
    assert site.source == "manual"

    # PreCompDataProtein: crashes or silently scores an empty pocket without these
    assert site.rdkit_lining_mol is not None
    assert site.rdkit_lining_mol.GetNumConformers() == 1
    assert site.lining_residues, "empty lining set scores a pocket with no protein"
    assert site.distance_map is not None and np.any(site.distance_map > PROBE)

    # _minimize_and_rescore reads `residues`, which is a different field
    assert site.residues

    # alpha_mask._centroid_in_existing_site uses a strict min < c < max test; left at
    # the (0,0,0) default it never matches, and every blob inside this site then spawns
    # a duplicate site -- the exact failure --manual-site exists to remove.
    assert np.allclose(site.min_coords, site.origin)
    assert np.all(np.asarray(site.min_coords) < centroid)
    assert np.all(np.asarray(site.max_coords) > centroid)

    assert site.site_centers.shape == (1, 3)
    assert site.site_radii.shape == (1,)
    assert site.volume > 0.0


def test_radius_mode_trims_the_circumscribing_cube_to_the_sphere(tmp_path):
    from ChemEM.tools.precomputed_data import (covert_idx_to_coords,
                                               get_valid_points_and_adjacency)

    centroid = np.array([0.0, 0.0, 0.0])
    radius = 6.0
    site = _build(_shell(centroid, 40.0), centroid, tmp_path,
                  radius=radius, grid_spacing=0.5)

    points, _ = get_valid_points_and_adjacency(site.distance_map > PROBE,
                                               connectivity=26.0)
    coords = covert_idx_to_coords(np.array(points), site.origin, site.apix)
    d = np.linalg.norm(coords - centroid, axis=1)

    # make_grid_and_origin_from_radius returns the CUBE around the sphere; the builder
    # must trim it, or the corners reach radius*sqrt(3)
    assert d.max() <= radius + 0.5, f"max {d.max():.2f} exceeds the requested {radius}"
    assert d.max() > radius - 1.0, "sphere trimmed far too aggressively"


def test_returns_none_when_the_volume_is_solid_protein(tmp_path):
    """Better a skipped site than a half-populated model that crashes downstream."""
    centroid = np.array([0.0, 0.0, 0.0])
    # a dense block straddling the whole requested box
    g = np.arange(-4.0, 4.01, 1.0)
    solid = np.array([[x, y, z] for x in g for y in g for z in g])

    assert _build(solid, centroid, tmp_path, box=[4.0], grid_spacing=0.5) is None


# --------------------------------------------------------------------------------------
# the --dock-full-map crop must be gated by the site's accessible mask
# --------------------------------------------------------------------------------------
def _map_stub(shape, origin, apix, fill=1.0):
    from ChemEM.parsers.EMMap import EMMap
    return EMMap(np.asarray(origin, dtype=float), tuple(apix),
                 np.full(shape, float(fill), dtype=np.float32), 2.0)


def _system_stub(*, manual_site, confidence_map, site_maps):
    import types
    return types.SimpleNamespace(
        options=types.SimpleNamespace(manual_site=manual_site, dock_full_map=True),
        confidence_map=confidence_map,
        binding_site_maps=site_maps,
        log=lambda *_a, **_k: None,
    )


def _site_stub(shape, origin, apix, accessible):
    return BindingSiteModel.from_dict({
        "key": 0, "source": "manual",
        "origin": np.asarray(origin, dtype=float), "apix": tuple(apix),
        "box_size": shape,
        "densmap": accessible.astype(np.float32),
        "distance_map": np.where(accessible, 10.0, 0.0).astype(np.float32),
    })


def test_dock_full_map_crop_is_masked_to_the_accessible_region(tmp_path):
    """Otherwise MI/SCI score protein density the ligand centroid can never occupy."""
    from ChemEM.tools.precomputed_data import resolve_docking_density_map

    shape, origin, apix = (6, 6, 6), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    accessible = np.zeros(shape, dtype=bool)
    accessible[1:4, 1:4, 1:4] = True                       # 27 of 216 voxels

    parent = _map_stub(shape, origin, apix, fill=1.0)       # dense everywhere
    segmented = _map_stub(shape, origin, apix, fill=0.0)
    site = _site_stub(shape, origin, apix, accessible)

    system = _system_stub(manual_site=True, confidence_map=parent,
                          site_maps={0: [(segmented, {})]})
    emmap, source = resolve_docking_density_map(system, site)

    assert "site-masked" in source
    assert int(np.count_nonzero(emmap.density_map)) == int(accessible.sum()) == 27
    assert np.all(emmap.density_map[~accessible] == 0.0)
    # the parent map must not be mutated by the masking
    assert np.count_nonzero(parent.density_map) == 216


def test_automatic_sites_keep_the_unmasked_full_map_crop(tmp_path):
    """--dock-full-map's documented behaviour is unchanged off the manual path."""
    from ChemEM.tools.precomputed_data import resolve_docking_density_map

    shape, origin, apix = (6, 6, 6), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    accessible = np.zeros(shape, dtype=bool)
    accessible[1:4, 1:4, 1:4] = True

    parent = _map_stub(shape, origin, apix, fill=1.0)
    segmented = _map_stub(shape, origin, apix, fill=0.0)
    site = _site_stub(shape, origin, apix, accessible)

    system = _system_stub(manual_site=False, confidence_map=parent,
                          site_maps={0: [(segmented, {})]})
    emmap, source = resolve_docking_density_map(system, site)

    assert "site-masked" not in source
    assert int(np.count_nonzero(emmap.density_map)) == 216


def test_mask_shape_mismatch_leaves_the_crop_alone(tmp_path):
    from ChemEM.tools.precomputed_data import resolve_docking_density_map

    shape, origin, apix = (6, 6, 6), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    parent = _map_stub(shape, origin, apix, fill=1.0)
    segmented = _map_stub(shape, origin, apix, fill=0.0)

    site = _site_stub(shape, origin, apix, np.ones((4, 4, 4), dtype=bool))
    system = _system_stub(manual_site=True, confidence_map=parent,
                          site_maps={0: [(segmented, {})]})

    emmap, source = resolve_docking_density_map(system, site)
    assert "site-masked" not in source
    assert int(np.count_nonzero(emmap.density_map)) == 216


def test_covalent_lining_atoms_are_forced_into_the_lining_set(tmp_path):
    centroid = np.array([0.0, 0.0, 0.0])
    coords = np.vstack([_shell(centroid, 40.0, n=10),
                        np.array([[60.0, 0.0, 0.0], [61.0, 0.0, 0.0]])])
    far = [len(coords) - 2, len(coords) - 1]     # nowhere near the site

    plain = _build(coords, centroid, tmp_path, box=[8.0], grid_spacing=0.5)
    forced = _build(coords, centroid, tmp_path, box=[8.0], grid_spacing=0.5,
                    extra_lining_atom_indices=far)

    assert len(forced.lining_residues) > len(plain.lining_residues)

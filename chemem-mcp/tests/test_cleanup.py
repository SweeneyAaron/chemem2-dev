from __future__ import annotations

from pathlib import Path

import pytest

from chemem_mcp.cleanup import UnsafeCleanupError, cleanup_run_dir


def _populate(run_dir: Path) -> None:
    """Create a realistic file tree under run_dir."""
    (run_dir / "inputs").mkdir()
    (run_dir / "inputs" / "model.pdb").write_text("ATOM\n")
    (run_dir / "inputs" / "map.mrc").write_bytes(b"MRC")

    (run_dir / "final").mkdir()
    (run_dir / "final" / "report.json").write_text("{}")
    (run_dir / "final" / "model.pdb").write_text("ATOM\n")

    (run_dir / "state.json").write_text("{}")
    (run_dir / "decision_log.jsonl").write_text('{"step":1}\n')
    (run_dir / "resource_usage.json").write_text("{}")

    (run_dir / "docking_scratch").mkdir()
    (run_dir / "docking_scratch" / "pose_001.tmp").write_bytes(b"x" * 1024)
    (run_dir / "docking_scratch" / "pose_002.tmp").write_bytes(b"x" * 2048)

    (run_dir / "derived_maps").mkdir()
    (run_dir / "derived_maps" / "crop_seg_001.mrc").write_bytes(b"x" * 512)

    (run_dir / "tmp").mkdir()
    (run_dir / "tmp" / "stuff.tmp").write_bytes(b"x" * 16)

    (run_dir / "scratch").mkdir()
    (run_dir / "scratch" / "anything").write_bytes(b"x" * 8)

    (run_dir / "poses").mkdir()
    (run_dir / "poses" / "ATP_pose_001.sdf").write_text("sdf")


def test_dry_run_reports_but_does_not_delete(tmp_path: Path) -> None:
    run_dir = tmp_path / "case001"
    run_dir.mkdir()
    _populate(run_dir)

    report = cleanup_run_dir(run_dir, dry_run=True).to_dict()
    assert report["status"] == "ok"
    assert report["dry_run"] is True
    assert report["deleted_file_count"] >= 5

    # nothing actually removed
    assert (run_dir / "docking_scratch" / "pose_001.tmp").exists()
    assert (run_dir / "derived_maps" / "crop_seg_001.mrc").exists()
    assert (run_dir / "tmp" / "stuff.tmp").exists()


def test_real_run_deletes_only_deletable(tmp_path: Path) -> None:
    run_dir = tmp_path / "case001"
    run_dir.mkdir()
    _populate(run_dir)

    cleanup_run_dir(run_dir, dry_run=False)

    # protected paths untouched
    assert (run_dir / "inputs" / "model.pdb").exists()
    assert (run_dir / "inputs" / "map.mrc").exists()
    assert (run_dir / "state.json").exists()
    assert (run_dir / "decision_log.jsonl").exists()
    assert (run_dir / "final" / "report.json").exists()
    assert (run_dir / "final" / "model.pdb").exists()
    assert (run_dir / "resource_usage.json").exists()

    # deletable categories gone
    assert not (run_dir / "docking_scratch" / "pose_001.tmp").exists()
    assert not (run_dir / "derived_maps" / "crop_seg_001.mrc").exists()
    assert not (run_dir / "tmp" / "stuff.tmp").exists()
    assert not (run_dir / "scratch" / "anything").exists()

    # neutral files (no rule matches) are left alone — poses are not in default delete globs
    assert (run_dir / "poses" / "ATP_pose_001.sdf").exists()


def test_inputs_protected_even_if_listed_in_delete_globs(tmp_path: Path) -> None:
    """User-supplied delete_globs cannot override always-keep patterns."""
    run_dir = tmp_path / "case001"
    run_dir.mkdir()
    _populate(run_dir)

    cleanup_run_dir(run_dir, dry_run=False, delete_globs=["**/*"])

    # built-in always-keep wins
    assert (run_dir / "inputs" / "model.pdb").exists()
    assert (run_dir / "state.json").exists()
    assert (run_dir / "decision_log.jsonl").exists()
    assert (run_dir / "final" / "report.json").exists()


def test_extra_keep_protects_additional_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "case001"
    run_dir.mkdir()
    _populate(run_dir)

    cleanup_run_dir(
        run_dir,
        dry_run=False,
        extra_keep=["poses/**", "docking_scratch/pose_001.tmp"],
    )
    assert (run_dir / "poses" / "ATP_pose_001.sdf").exists()
    assert (run_dir / "docking_scratch" / "pose_001.tmp").exists()
    # the un-protected scratch file still goes
    assert not (run_dir / "docking_scratch" / "pose_002.tmp").exists()


def test_missing_run_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(UnsafeCleanupError):
        cleanup_run_dir(tmp_path / "does_not_exist", dry_run=True)


def test_freed_bytes_accounted(tmp_path: Path) -> None:
    run_dir = tmp_path / "case001"
    run_dir.mkdir()
    _populate(run_dir)
    report = cleanup_run_dir(run_dir, dry_run=True).to_dict()
    # docking_scratch alone is 1024+2048=3072 bytes; tmp+scratch+crop add more
    assert report["freed_bytes"] >= 1024 + 2048 + 512 + 16 + 8

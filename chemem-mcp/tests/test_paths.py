from __future__ import annotations

from pathlib import Path

import pytest

from chemem_mcp.paths import UnsafePathError, safe_child, safe_run_dir, validate_run_id


def test_safe_run_dir_creates_dir(tmp_path: Path) -> None:
    run_dir = safe_run_dir("case001", tmp_path)
    assert run_dir.exists()
    assert run_dir.is_dir()
    assert run_dir == (tmp_path / "case001").resolve()


def test_safe_run_dir_idempotent(tmp_path: Path) -> None:
    a = safe_run_dir("case001", tmp_path)
    b = safe_run_dir("case001", tmp_path)
    assert a == b


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "..",
        "../escape",
        "case/with/slash",
        "/absolute",
        ".hidden",
        "a" * 100,
        "case 001",
        "case\x00null",
    ],
)
def test_safe_run_dir_rejects_unsafe(tmp_path: Path, bad: str) -> None:
    with pytest.raises(UnsafePathError):
        safe_run_dir(bad, tmp_path)


def test_validate_run_id_accepts_valid() -> None:
    for ok in ("a", "case_001", "Run-2", "x.y", "A1.b-c_d"):
        validate_run_id(ok)


def test_safe_child_rejects_escape(tmp_path: Path) -> None:
    parent = safe_run_dir("case001", tmp_path)
    with pytest.raises(UnsafePathError):
        safe_child(parent, "..", "escape.txt")


def test_safe_child_ok(tmp_path: Path) -> None:
    parent = safe_run_dir("case001", tmp_path)
    child = safe_child(parent, "inputs", "model.pdb")
    assert child == (parent / "inputs" / "model.pdb").resolve()

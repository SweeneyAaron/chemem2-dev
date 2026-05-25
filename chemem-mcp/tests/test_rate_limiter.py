from __future__ import annotations

import pytest

from chemem_mcp.rate_limiter import (
    DEFAULT_FALLBACK_LIMIT,
    DEFAULT_LIMITS,
    RateLimiter,
)


def test_check_and_increment_allows_until_limit() -> None:
    rl = RateLimiter(limits={"tool_a": 3})
    for i in range(3):
        res = rl.check_and_increment("tool_a", "case001:k")
        assert res.allowed
        assert res.used == i + 1
        assert res.remaining == 2 - i


def test_check_and_increment_blocks_after_limit() -> None:
    rl = RateLimiter(limits={"tool_a": 2})
    rl.check_and_increment("tool_a", "case001:k")
    rl.check_and_increment("tool_a", "case001:k")
    res = rl.check_and_increment("tool_a", "case001:k")
    assert res.allowed is False
    assert res.used == 2
    assert res.remaining == 0


def test_peek_does_not_increment() -> None:
    rl = RateLimiter(limits={"tool_a": 2})
    res_before = rl.peek("tool_a", "k")
    assert res_before.used == 0
    rl.peek("tool_a", "k")
    rl.peek("tool_a", "k")
    res_after = rl.peek("tool_a", "k")
    assert res_after.used == 0
    assert res_after.allowed


def test_keys_are_independent() -> None:
    rl = RateLimiter(limits={"tool_a": 1})
    a = rl.check_and_increment("tool_a", "case001:k1")
    b = rl.check_and_increment("tool_a", "case001:k2")
    assert a.allowed and b.allowed


def test_fallback_limit_used_for_unknown_tool() -> None:
    rl = RateLimiter(limits={}, fallback_limit=4)
    assert rl.get_limit("unknown_tool") == 4


def test_with_defaults_has_known_tools() -> None:
    rl = RateLimiter.with_defaults()
    # spot-check a few tools from the plan's table
    assert rl.get_limit("dock_ligands_to_segment") == 3
    assert rl.get_limit("run_smart_refine_2") == 5
    assert rl.get_limit("qwen_controller_step") == 100
    # unknown still falls back
    assert rl.get_limit("not_a_tool") == DEFAULT_FALLBACK_LIMIT


def test_summary_filters_by_run_id() -> None:
    rl = RateLimiter(limits={"tool_a": 5, "tool_b": 5})
    rl.check_and_increment("tool_a", "case001:k1")
    rl.check_and_increment("tool_b", "case001:k2")
    rl.check_and_increment("tool_a", "case002:k1")
    summary = rl.summary(run_id="case001")
    keys = {r["key"] for r in summary["rate_limits"]}
    assert keys == {"case001:k1", "case001:k2"}


def test_summary_all_when_no_run_id() -> None:
    rl = RateLimiter(limits={"tool_a": 5})
    rl.check_and_increment("tool_a", "case001:k1")
    rl.check_and_increment("tool_a", "case002:k1")
    summary = rl.summary()
    assert len(summary["rate_limits"]) == 2


def test_reset_specific_tool() -> None:
    rl = RateLimiter(limits={"tool_a": 5, "tool_b": 5})
    rl.check_and_increment("tool_a", "k")
    rl.check_and_increment("tool_b", "k")
    removed = rl.reset(tool_name="tool_a")
    assert removed == 1
    assert rl.peek("tool_a", "k").used == 0
    assert rl.peek("tool_b", "k").used == 1


def test_reset_all() -> None:
    rl = RateLimiter(limits={"tool_a": 5})
    rl.check_and_increment("tool_a", "k1")
    rl.check_and_increment("tool_a", "k2")
    assert rl.reset() == 2
    assert rl.summary()["rate_limits"] == []


def test_default_limits_table_complete() -> None:
    # Sanity: a curated subset of plan-block tools must have explicit defaults.
    expected = {
        "prepare_case",
        "generate_alpha_masks",
        "dock_ligands_to_segment",
        "run_smart_refine_2",
        "cleanup_intermediate_files",
        "increase_docking_diversity",
        "loosen_alpha_mask",
    }
    missing = expected - DEFAULT_LIMITS.keys()
    assert not missing, f"missing default rate limits: {missing}"

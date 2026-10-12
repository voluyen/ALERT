import pytest
import torch

from answergain.allocation.weights import no_answer_upweight_step_weights, step_weights
from answergain.signals.predictability import step_mean_log_probabilities, token_log_probabilities
from answergain.signals.score_transform import transform_scores


def test_no_answer_upweight_pins_answer_steps_and_preserves_trace_mass():
    lengths = [4, 2, 6, 3]
    answer_only = [False, False, True, False]
    weights = no_answer_upweight_step_weights([0.3, 1.2, 9.0, -0.5], lengths, answer_only, 2.0, 0.5)
    assert weights[2] == 1.0
    assert sum(w * n for w, n in zip(weights, lengths)) == pytest.approx(sum(lengths))
    eligible = [0, 1, 3]
    assert sum(weights[i] * lengths[i] for i in eligible) == pytest.approx(sum(lengths[i] for i in eligible))


def test_no_answer_upweight_ignores_answer_only_scores():
    lengths, labels = [3, 5, 2], [False, True, False]
    a = no_answer_upweight_step_weights([0.1, 100.0, 0.9], lengths, labels, 2.0, 0.5)
    b = no_answer_upweight_step_weights([0.1, -100.0, 0.9], lengths, labels, 2.0, 0.5)
    assert a == pytest.approx(b)


def test_no_answer_upweight_all_answer_only_is_uniform():
    assert no_answer_upweight_step_weights([1.0, 2.0], [3, 4], [True, True]) == [1.0, 1.0]


def test_no_answer_upweight_without_labels_matches_allocation():
    scores, lengths = [0.2, 0.9, 0.4], [2, 5, 3]
    got = no_answer_upweight_step_weights(scores, lengths, [False] * 3, 2.0, 0.5, 2.0)
    assert got == pytest.approx(step_weights(scores, lengths, 2.0, 0.5, 2.0))


def test_no_answer_upweight_rejects_misaligned_labels():
    with pytest.raises(ValueError):
        no_answer_upweight_step_weights([1.0, 2.0], [1, 1], [False])


def test_negate_flips_sign():
    out, perms = transform_scores({"1": [0.5, -2.0]}, "negate")
    assert out == {"1": [-0.5, 2.0]} and perms is None


def test_shuffle_is_a_recorded_deterministic_permutation():
    scores = {"3": [0.1, 0.2, 0.3, 0.4, 0.5], "4": [1.0, 2.0, 3.0]}
    out, perms = transform_scores(scores, "shuffle", seed=42)
    again, _ = transform_scores(scores, "shuffle", seed=42)
    assert out == again
    for key, values in scores.items():
        assert sorted(out[key]) == sorted(values)
        assert out[key] == [values[i] for i in perms[key]]


def test_transform_rejects_non_finite_and_unknown_mode():
    with pytest.raises(ValueError):
        transform_scores({"1": [float("nan")]}, "negate")
    with pytest.raises(ValueError):
        transform_scores({"1": [1.0]}, "bogus")


def test_chunked_log_probabilities_match_full_log_softmax():
    torch.manual_seed(0)
    hidden, unembed = torch.randn(7, 8), torch.randn(11, 8)
    targets = torch.randint(0, 11, (7,))
    expected = torch.log_softmax(hidden @ unembed.T, dim=-1).gather(1, targets[:, None]).squeeze(1)
    assert torch.allclose(token_log_probabilities(hidden, targets, unembed, chunk_size=3), expected, atol=1e-5)


def test_step_mean_uses_response_relative_spans():
    logps = torch.tensor([-1.0, -3.0, -2.0, -4.0])
    assert step_mean_log_probabilities(logps, [(10, 12), (12, 14)], 10) == pytest.approx([-2.0, -3.0])


def _write_run(results, tag, p1, p3):
    import json
    rows = [{"benchmark": name, "pass@1": p1, "pass@3": p3, "model": tag}
            for name in ("aime24", "aime25", "amc12", "math500")]
    (results / tag).mkdir(parents=True)
    (results / tag / "summary.json").write_text(json.dumps(rows))


def test_summary_collects_main_table_seeds_and_missing_runs(tmp_path):
    from answergain.eval.answer_gain_summary import main

    results, out, track = tmp_path / "results", tmp_path / "out", "r1-qwen-1.5b-palign"
    _write_run(results, f"e0-uniform-{track}", 0.30, 0.40)
    _write_run(results, f"e1-alg-{track}", 0.35, 0.45)
    _write_run(results, f"e1-alg-{track}-s123", 0.37, 0.47)
    main(["--results-dir", str(results), "--out-dir", str(out), "--data-dir", str(tmp_path)])

    report = (out / "ablation_summary.md").read_text()
    assert "| E1 ALG |" in report and "35.00%" in report
    assert f"`e2-predictability-easy-{track}`" in report  # not run -> listed as missing
    assert "36.00% ± " in report                           # E1 seeds 42 and 123
    assert (out / "per_benchmark.csv").read_text().count("\n") == 4 * 3 + 1
    assert "e1-alg" in (out / "seed_summary.csv").read_text()

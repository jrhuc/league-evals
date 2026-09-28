import pytest

from league_evals.analysis import pair_mean, paired_contrast


def cell(seed=1, win=1, complete=True, team="b"):
    return {
        "focal_team": "a",
        "opponent_team": team,
        "focal_seat": "p1",
        "seed": seed,
        "epoch": 1,
        "pair": f"a--{team}",
        "complete": complete,
        "win": win,
    }


def test_pair_weighting_does_not_count_repeats_as_new_pairs():
    rows = [{"pair": "a--b", "win": 1}] * 20 + [{"pair": "a--c", "win": 0}]
    assert pair_mean(rows) == {"estimate": 0.5, "pairs": 2, "by_pair": {"a--b": 1, "a--c": 0}}
    assert pair_mean([])["estimate"] is None


def test_complete_comparison_reports_discordance_and_exact_bounds():
    left = [cell(win=0), cell(seed=2), cell(team="c")]
    right = [cell(), cell(seed=2), cell(team="c", win=0)]
    result = paired_contrast(left, right)
    assert result["matched_cells"] == result["observed_cells"] == 3
    assert result["right_only_wins"] == result["left_only_wins"] == result["same_outcome"] == 1
    assert result["estimate"] == -0.25  # mean(pair b: +0.5, pair c: -1)
    assert result["missing_outcome_bounds"] == [-0.25, -0.25]


def test_incomplete_and_missing_cells_can_reverse_the_complete_case_conclusion():
    result = paired_contrast(
        [cell(win=0), cell(seed=2), cell(seed=3, complete=False)],
        [cell(), cell(seed=2, complete=False), cell(seed=4, complete=False)],
    )
    assert result["estimate"] == 1
    assert result["matched_cells"] == 1
    assert result["observed_cells"] == 4
    assert result["left_missing"] == result["right_missing"] == 1
    assert result["left_incomplete"] == 1
    assert result["right_incomplete"] == 2
    assert result["missing_outcome_bounds"] == [-0.5, 0.75]


@pytest.mark.parametrize("rows", [[cell(), cell()], [cell(complete=False), cell()]])
def test_duplicate_attempts_cannot_be_hidden_by_dropping_failures(rows):
    with pytest.raises(ValueError, match="duplicate"):
        paired_contrast(rows, [cell()])


def test_retried_samples_are_not_valid_single_attempt_comparisons():
    with pytest.raises(ValueError, match="retries"):
        paired_contrast([{**cell(), "retried": True}], [cell()])


def test_no_completed_matches_returns_unknown_estimate_with_full_bounds():
    result = paired_contrast([cell(complete=False)], [cell(complete=False)])
    assert result["estimate"] is None
    assert result["missing_outcome_bounds"] == [-1, 1]
    assert paired_contrast([], [])["missing_outcome_bounds"] == [None, None]


def test_incomplete_metadata_is_not_silently_matched():
    with pytest.raises(ValueError, match="metadata"):
        paired_contrast([{**cell(), "seed": None}], [cell()])

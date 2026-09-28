import math

from league_evals.scorers import (
    discipline_summary,
    efficiency_summary,
    mechanics_summary,
    outcome_summary,
)


def test_outcome_summary_counts_focal_win_and_rejections():
    outcome = {
        "winner": "focal",
        "turns": 9,
        "simulator_substitutions": {"p1": 0, "p2": 0},
        "submissions": {
            "p1": [
                {"outcome": "accepted"},
                {"outcome": "rejected"},
                {"outcome": "accepted"},
            ],
            "p2": [],
        },
    }
    summary = outcome_summary(outcome)
    assert summary["win"] == 1.0
    assert summary["turns"] == 9
    assert summary["showdown_rejections"] == 1


def test_mechanics_summary_leaves_rates_undefined_without_predictions():
    empty = {
        "damagePredictions": 0,
        "damageMatched": 0,
        "orderPredictions": 0,
        "orderMatched": 0,
        "findings": [],
    }
    assert math.isnan(mechanics_summary(empty)["damage_pair_rate"])
    assert math.isnan(mechanics_summary(empty)["findings_per_match"])
    audit = {
        "damagePredictions": 6,
        "damageMatched": 3,
        "damageHypothetical": 2,
        "orderPredictions": 2,
        "orderMatched": 2,
        "orderHypothetical": 0,
        "findings": [{"detail": "x"}],
    }
    summary = mechanics_summary(audit)
    assert summary["damage_pair_rate"] == 0.75
    assert summary["order_pair_rate"] == 1.0
    assert summary["hypothetical_predictions"] == 2
    assert summary["findings"] == 1
    assert summary["findings_per_match"] == 0.2


def test_discipline_summary_separates_preview_calcs_from_lookups():
    decisions = [
        {
            "phase": "team_preview",
            "calls": ["estimate_damage", "estimate_damage", "lookup_species"],
            "generations": 2,
            "rejected": 0,
            "defaulted": False,
        },
        {"phase": "turn", "calls": [], "generations": 1, "rejected": 1, "defaulted": False},
        {
            "phase": "turn",
            "calls": ["compare_action_order"],
            "generations": 3,
            "rejected": 0,
            "defaulted": True,
        },
    ]
    summary = discipline_summary(decisions)
    assert summary["decisions"] == 3
    assert summary["preview_calcs"] == 2
    assert summary["calc_decision_rate"] == 2 / 3
    assert summary["tool_calls_per_decision"] == 4 / 3
    assert summary["generations_per_decision"] == 2.0
    assert summary["rejected_submissions"] == 1
    assert summary["defaulted_decisions"] == 1


def test_efficiency_summary_handles_zero_decisions():
    assert math.isnan(efficiency_summary(1000, 0)["tokens_per_decision"])
    assert efficiency_summary(1000, 4)["tokens_per_decision"] == 250


def test_p2_substitutions_and_assisted_wins_are_not_credited_as_unassisted():
    result = {
        "winner": "focal",
        "turns": 3,
        "simulator_substitutions": {"p1": 0, "p2": 1},
        "submissions": {"p1": [], "p2": [{"outcome": "accepted", "source": "model"}]},
    }
    summary = outcome_summary(result, "p2")
    assert summary["win"] == 1
    assert summary["unassisted_win"] == 0
    assert summary["assisted"] == 1
    assert summary["simulator_substitutions"] == 1
    assert outcome_summary(result, "p1", defaulted=True)["unassisted_win"] == 0


def test_multiple_findings_per_match_are_diagnostic_density_not_probability():
    summary = mechanics_summary(
        {
            "damagePredictions": 1,
            "damageMatched": 1,
            "orderPredictions": 0,
            "orderMatched": 0,
            "findings": [{"detail": "range"}, {"detail": "ko"}],
        }
    )
    assert summary["findings_per_match"] == 2

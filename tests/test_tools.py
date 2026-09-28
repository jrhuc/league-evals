from league_evals.battle import battle_samples, tool_params

SCHEMA = {
    "type": "object",
    "properties": {
        "attacker": {"type": "string", "description": "Attacking species."},
        "helping_hand": {"type": "boolean"},
        "attacker_stats": {
            "type": "object",
            "properties": {"atk": {"type": "number", "exclusiveMinimum": 0}},
            "additionalProperties": False,
        },
    },
    "required": ["attacker", "not_a_property"],
    "additionalProperties": False,
}


def test_tool_params_keeps_optional_arguments_optional():
    params = tool_params(SCHEMA)
    dumped = params.model_dump(exclude_none=True)
    assert set(dumped["properties"]) == {"attacker", "helping_hand", "attacker_stats"}
    assert dumped["required"] == ["attacker"]
    nested = dumped["properties"]["attacker_stats"]
    assert "additionalProperties" not in nested
    assert nested["properties"]["atk"] == {"type": "number"}


def test_battle_samples_cover_every_ordered_pair_per_seed():
    pool = {
        "format": "gen9championsvgc2026regmcbo3",
        "teams": [
            {"id": "a", "packed": "A"},
            {"id": "b", "packed": "B"},
            {"id": "c", "packed": "C"},
        ],
    }
    samples = battle_samples(pool, [1, 2], opponent="random")
    assert len(samples) == 24
    first = samples[0]
    assert first.id == "a--b--1--p1"
    assert first.metadata["focal_packed"] == "A"
    assert first.metadata["opponent_packed"] == "B"
    assert first.metadata["opponent_policy"] == "random"
    assert first.metadata["pair"] == "a--b"
    assert {s.metadata["pair"] for s in samples} == {"a--b", "a--c", "b--c"}

    assert {s.metadata["focal_seat"] for s in samples} == {"p1", "p2"}


def test_combined_experiment_has_identical_cells_in_both_conditions():
    from league_evals.battle import vgc_battle

    samples = list(vgc_battle(tool_access="both").dataset)
    assert len(samples) == len({s.id for s in samples}) == 120
    fields = ("focal_team", "opponent_team", "focal_seat", "seed")
    cells = {
        condition: {
            tuple(s.metadata[k] for k in fields)
            for s in samples
            if s.metadata["tool_access"] == condition
        }
        for condition in ("full", "no_calculators")
    }
    assert cells["full"] == cells["no_calculators"]
    assert len(cells["full"]) == 60

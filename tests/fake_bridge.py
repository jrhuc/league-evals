"""Bridge messages in the shape `vgcleague bridge` sends, for tests that run without Node."""

SEATS = ["external", "random", "greedy", "search", "search:fast"]
SYSTEM = "You are an expert VGC player."
SUBMISSION = {
    "name": "submit_action",
    "description": "Submit your decision.",
    "parameters": {
        "type": "object",
        "properties": {
            "choices": {"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 0}},
            "rationale": {"type": "string", "maxLength": 2000},
            "notebook": {
                "type": "object",
                "properties": {"team_playbook": {"type": "string"}},
                "additionalProperties": False,
            },
        },
        "required": ["choices"],
        "additionalProperties": False,
    },
}


def tools(*names):
    return [
        {
            "name": name,
            "description": "Reference lookup.",
            "parameters": {"type": "object", "properties": {}},
        }
        for name in names
    ]


def external(start):
    return next(pid for pid in ("p1", "p2") if start[pid]["seat"] == "external")


def exchange_event(
    pid, number, *, turn, phase="turn", prompt="Choose index 0.", view=None, names=()
):
    return {
        "kind": "exchange",
        "pid": pid,
        "exchange": {
            "id": number,
            "session": f"battle-eval-{pid}",
            "task": f"decision-{number}",
            "system": SYSTEM,
            "prompt": prompt,
            "tools": tools(*names),
            "submission": SUBMISSION,
        },
        "decision": {
            "turn": turn,
            "phase": phase,
            "slot_names": [],
            "menus": [],
            "request": {},
            **(view or {}),
        },
    }


def decision_row(pid, action, *, source="model", outcome="accepted", error=None):
    return {
        "kind": "decision",
        "pid": pid,
        "row": {
            "kind": "decision",
            "pid": pid,
            "action": action,
            "automatic": source == "automatic",
            "submission_source": source,
            "outcome": outcome,
            "showdown_error": error,
        },
    }

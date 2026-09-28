"""Inspect mock provider with explicit usage, so tests never download tokenizers."""

from inspect_ai.model import ModelOutput, ModelUsage, get_model


def offline_model(outputs=None):
    iterator = iter(outputs) if outputs is not None and not callable(outputs) else None

    def respond(*args):
        if callable(outputs):
            output = outputs(*args)
        elif iterator is not None:
            output = next(iterator)
        else:
            output = ModelOutput.from_content("mock", "No action submitted.")
        output.usage = ModelUsage(input_tokens=10, output_tokens=10, total_tokens=20)
        return output

    return get_model("mockllm/model", custom_outputs=respond)

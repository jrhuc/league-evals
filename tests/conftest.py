"""Keep Inspect's logs/cache inside the test directory, including on developer machines."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_inspect_state(tmp_path_factory):
    from inspect_ai._util import appdirs

    root = tmp_path_factory.mktemp("inspect-state")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(appdirs, "user_data_path", lambda package: root / "data" / package)
        patch.setattr(appdirs, "user_cache_path", lambda package: root / "cache" / package)
        yield

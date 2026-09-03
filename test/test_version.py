import tomllib
from pathlib import Path

import jaxdp


def test_version_matches_package_metadata() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert jaxdp.__version__ == metadata["project"]["version"]

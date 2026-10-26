"""
The code examples in the README must run (and keep running when the API changes).
"""

import re
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

README = Path(__file__).resolve().parents[1] / "README.md"


def _python_blocks():
    text = README.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    # complete programs only (fragments such as a single compile() call have no import)
    return [b for b in blocks if "import" in b]


def test_readme_has_examples():
    assert len(_python_blocks()) >= 2


@pytest.mark.parametrize("index", range(len(_python_blocks())))
def test_readme_example_runs(index, monkeypatch, capsys):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    exec(compile(_python_blocks()[index], f"README block {index}", "exec"), {"__name__": "__main__"})
    plt.close("all")
    assert "scores" in capsys.readouterr().out  # every example prints its scores

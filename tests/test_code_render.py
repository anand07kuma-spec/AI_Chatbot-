"""Runs the JS code-rendering regression test (tests/code_render_test.js).

Guards the fix for HTML syntax-highlight tokens leaking as literal text
inside rendered code blocks (sequential highlight passes re-scanned their
own <span class="..."> output and corrupted it).
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tests" / "code_render_test.js"
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None or not SCRIPT.exists(), reason="node or test script unavailable")
def test_code_render_js():
    proc = subprocess.run(
        [NODE, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "assertions groups passed" in proc.stdout, proc.stdout

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PaperEntrypointTests(unittest.TestCase):
    def test_documented_python_entrypoints_compile(self):
        with tempfile.TemporaryDirectory() as cache_directory:
            environment = os.environ.copy()
            environment["PYTHONPYCACHEPREFIX"] = cache_directory
            for relative_path in ("gd_pops_v8.py", "gd_pops_v10.py"):
                subprocess.run(
                    [sys.executable, "-m", "py_compile", str(REPOSITORY_ROOT / relative_path)],
                    check=True,
                    env=environment,
                )

    def test_documented_shell_entrypoints_parse(self):
        for relative_path in (
            "gd_pops_v8_task.sh",
            "run_synth_expt.sh",
            "run_uci_expt.sh",
            "run_acs_expt.sh",
        ):
            subprocess.run(
                ["bash", "-n", str(REPOSITORY_ROOT / relative_path)],
                check=True,
            )


if __name__ == "__main__":
    unittest.main()

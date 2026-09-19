import os
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from run_experiment import restore_saved_args_for_resume


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PaperEntrypointTests(unittest.TestCase):
    def test_documented_python_entrypoints_compile(self):
        with tempfile.TemporaryDirectory() as cache_directory:
            environment = os.environ.copy()
            environment["PYTHONPYCACHEPREFIX"] = cache_directory
            for relative_path in ("run_experiment.py",):
                subprocess.run(
                    [sys.executable, "-m", "py_compile", str(REPOSITORY_ROOT / relative_path)],
                    check=True,
                    env=environment,
                )

    def test_documented_shell_entrypoints_parse(self):
        for relative_path in (
            "run_experiment_task.sh",
            "run_synth_expt.sh",
            "run_uci_expt.sh",
            "run_acs_expt.sh",
        ):
            subprocess.run(
                ["bash", "-n", str(REPOSITORY_ROOT / relative_path)],
                check=True,
            )

    def test_resume_recovers_saved_arguments_without_starting_a_new_run(self):
        with tempfile.TemporaryDirectory() as run_directory:
            params = {
                "resume": None,
                "populations": ["linear_regression", "linear_regression"],
            }
            with open(Path(run_directory) / "experiment_params.json", "w") as params_file:
                json.dump(params, params_file)

            args = SimpleNamespace(resume=run_directory, populations=None)
            restored = restore_saved_args_for_resume(args)

            self.assertEqual(restored.resume, run_directory)
            self.assertEqual(
                restored.populations,
                ["linear_regression", "linear_regression"],
            )


if __name__ == "__main__":
    unittest.main()

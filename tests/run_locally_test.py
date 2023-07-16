"""Testing integration of components locally."""

import gc
import os
import tempfile

from prompt2model.run_locally import run_skeleton


def test_integration():
    """Check that a end-to-end run with a single prompt doesn't throw an error."""
    prompt = ["Test prompt"]
    with tempfile.TemporaryDirectory() as tmpdirname:
        metrics_output_path = os.path.join(tmpdirname, "metrics.json")
        run_skeleton(prompt, metrics_output_path)
    gc.collect()

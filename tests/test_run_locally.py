"""Testing integration of components locally."""

from prompt2model.run_locally import run_skeleton


def test_integration():
    """Check that a end-to-end run with a single prompt doesn't throw an error."""
    prompt = ["Test prompt"]
    run_skeleton(prompt, "")

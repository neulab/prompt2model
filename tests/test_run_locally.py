"""Testing integration of components locally."""

import os
import sys

sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../prompt2model'))
)

from run_locally import main


def test_integration():
    """Check that a end-to-end run with a single prompt doesn't throw an error.
    """
    prompt = ["Test prompt"]
    main(prompt)

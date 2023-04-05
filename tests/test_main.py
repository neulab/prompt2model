import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prompt2model import main

def test_integration():
    prompt = ["Test prompt"]
    main(prompt)
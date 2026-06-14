import unittest

from src.eval.code_eval import clean_generated_code, evaluate_functional_correctness
from src.prompts import code_prompt, doc_prompt


class TestPrompts(unittest.TestCase):
    def test_code_prompt_docstring(self):
        p = code_prompt("Write a function to add two numbers")
        self.assertIn('"""', p)
        self.assertIn("add two numbers", p)

    def test_doc_prompt(self):
        p = doc_prompt("def add(a,b): pass")
        self.assertIn("# Description:", p)
        self.assertIn("def add", p)


class TestCodeEval(unittest.TestCase):
    def test_clean_generated_code_truncates_second_def(self):
        raw = "def a():\n    return 1\n\ndef b():\n    return 2"
        cleaned = clean_generated_code(raw)
        self.assertIn("def a", cleaned)
        self.assertNotIn("def b", cleaned)

    def test_functional_correctness_pass(self):
        code = "def add(a, b):\n    return a + b"
        tests = ["assert add(1, 2) == 3"]
        self.assertTrue(evaluate_functional_correctness(code, tests))

    def test_functional_correctness_fail(self):
        code = "def add(a, b):\n    return a - b"
        tests = ["assert add(1, 2) == 3"]
        self.assertFalse(evaluate_functional_correctness(code, tests))


if __name__ == "__main__":
    unittest.main()

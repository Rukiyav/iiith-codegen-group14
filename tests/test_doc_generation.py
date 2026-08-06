import unittest

from src.generation.doc_postprocess import (
    clean_generated_documentation,
    documentation_to_comments,
    fallback_documentation,
    finalize_documentation,
)


class TestDocPostprocess(unittest.TestCase):
    def test_clean_strips_code_continuation(self):
        raw = "# Returns sum\n\ndef other():\n    pass"
        self.assertNotIn("def other", clean_generated_documentation(raw))

    def test_clean_strips_urls(self):
        raw = "http://stackoverflow.com/questions/123\n# Some text"
        cleaned = clean_generated_documentation(raw)
        self.assertNotIn("stackoverflow", cleaned)

    def test_documentation_to_comments(self):
        doc = '"""Sum values."""\n'
        out = documentation_to_comments(doc)
        self.assertTrue(out.startswith("#"))

    def test_fallback_from_ast(self):
        code = "def sum_of_n(n):\n    return n"
        out = fallback_documentation(code)
        self.assertIn("sum of n", out.lower())
        self.assertIn("Args:", out)

    def test_finalize_uses_fallback_on_code_spill(self):
        code = "def f(x):\n    return x"
        raw = "http://stackoverflow.com/foo\ndef f2(): pass"
        out = finalize_documentation(code, raw)
        self.assertNotIn("stackoverflow", out)
        self.assertNotIn("def f2", out)

    def test_finalize_rejects_leetcode_boilerplate(self):
        code = """def sum_of_n_natural_numbers(n):
    sum = 0
    for i in range(1, n + 1):
        sum += i
    return sum"""
        raw = "# Given an integer n\n# Companies\n# Google"
        out = finalize_documentation(code, raw)
        self.assertIn("1 to n", out)
        self.assertNotIn("Companies", out)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

from src.agent.code_agent import agent_generate_code
from src.data.mbpp import load_mbpp_raw, _normalize_example


class TestMbppNotebookParity(unittest.TestCase):
    def test_normalize_has_notebook_fields(self):
        ex = {
            "task_id": 11,
            "prompt": "Write a function",
            "code": "def f(): pass",
            "test_list": ["assert True"],
        }
        norm = _normalize_example(ex)
        self.assertEqual(norm["task_id"], 11)
        self.assertEqual(norm["code"], "def f(): pass")

    def test_load_mbpp_raw_count(self):
        raw = load_mbpp_raw()
        self.assertGreaterEqual(len(raw), 400)
        self.assertIn("prompt", raw[0])
        self.assertIn("test_list", raw[0])


class TestAgent(unittest.TestCase):
    @patch("src.agent.code_agent.generate_python_code")
    def test_agent_succeeds_on_retry(self, mock_gen):
        mock_gen.side_effect = [
            "def add(a,b):\n    return a-b",
            "def add(a,b):\n    return a+b",
        ]
        result = agent_generate_code(
            "add two numbers",
            ["assert add(1, 2) == 3"],
            max_retries=3,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(result.steps), 2)


if __name__ == "__main__":
    unittest.main()

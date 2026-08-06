import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import app


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    @patch("app.main.generate_python_code", return_value="def add(a,b): return a+b")
    def test_generate_code(self, _mock):
        resp = self.client.post(
            "/generate/code",
            json={"prompt": "add two numbers", "model": "codegen-350m"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("add", resp.json()["output"])

    def test_list_models(self):
        resp = self.client.get("/models")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("models", body)
        ids = {m["id"] for m in body["models"]}
        self.assertIn("codegen-350m", ids)

    @patch("app.main.agent_generate_code")
    def test_generate_code_with_agent(self, mock_agent):
        from src.agent.code_agent import AgentResult, AgentStep

        mock_agent.return_value = AgentResult(
            final_code="def add(a,b): return a+b",
            passed=True,
            attempts=2,
            steps=[AgentStep(1, "bad", False, "p"), AgentStep(2, "def add(a,b): return a+b", True, "p")],
        )
        resp = self.client.post(
            "/generate/code",
            json={
                "prompt": "add",
                "test_list": ["assert add(1,2)==3"],
                "use_agent": True,
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["passed"])
        self.assertEqual(len(body["steps"]), 2)

    def test_evaluate_code(self):
        resp = self.client.post(
            "/evaluate/code",
            json={
                "generated_code": "def add(a,b):\n    return a+b",
                "test_list": ["assert add(1,2)==3"],
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["passed"])


if __name__ == "__main__":
    unittest.main()

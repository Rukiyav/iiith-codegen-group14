import unittest

from src.models.registry import (
    DEFAULT_MODEL_ID,
    list_model_options,
    resolve_hub_path,
    resolve_model_spec,
)


class TestModelRegistry(unittest.TestCase):
    def test_presets_listed(self):
        ids = {m["id"] for m in list_model_options(include_checkpoints=False)}
        self.assertEqual(ids, {"codegen-350m"})

    def test_resolve_preset(self):
        self.assertEqual(
            resolve_hub_path("codegen-350m"),
            "Salesforce/codegen-350M-multi",
        )

    def test_default_spec(self):
        spec = resolve_model_spec(None)
        assert spec is not None
        self.assertEqual(spec.id, DEFAULT_MODEL_ID)


if __name__ == "__main__":
    unittest.main()

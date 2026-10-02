import unittest
from unittest import mock

from services.comfy_lora_resolver import (
    ComfyLoraResolutionError,
    _extract_lora_candidates,
    normalize_lora_name,
    resolve_comfy_lora_names,
)


class ComfyLoraResolverTestCase(unittest.TestCase):
    def test_extracts_lora_loader_candidates(self):
        payload = {
            "LoraLoader": {
                "input": {
                    "required": {
                        "lora_name": [[r"folder\style.safetensors", "flat.safetensors"]]
                    }
                }
            }
        }
        self.assertEqual(
            _extract_lora_candidates(payload),
            (r"folder\style.safetensors", "flat.safetensors"),
        )

    def test_catalog_forward_slash_resolves_to_exact_comfy_candidate(self):
        accepted = (r"こまごめピペット\komagome_style_v1.safetensors",)
        self.assertEqual(
            resolve_comfy_lora_names(
                ("こまごめピペット/komagome_style_v1.safetensors",),
                candidates=accepted,
            ),
            accepted,
        )

    def test_exact_candidate_is_preserved(self):
        accepted = ("style.safetensors",)
        self.assertEqual(resolve_comfy_lora_names(accepted, candidates=accepted), accepted)

    def test_separator_and_case_normalization_are_comparison_only(self):
        self.assertEqual(normalize_lora_name(r"./folder\\style.safetensors"), "folder/style.safetensors")
        accepted = (r"Folder\Style.safetensors",)
        self.assertEqual(
            resolve_comfy_lora_names(("folder/style.safetensors",), candidates=accepted),
            accepted,
        )

    def test_missing_candidate_fails_before_any_submission(self):
        with self.assertRaises(ComfyLoraResolutionError):
            resolve_comfy_lora_names(("missing.safetensors",), candidates=("known.safetensors",))

    def test_multiple_loras_resolve_independently(self):
        accepted = (r"one\a.safetensors", r"two\b.safetensors")
        self.assertEqual(
            resolve_comfy_lora_names(("one/a.safetensors", "two/b.safetensors"), candidates=accepted),
            accepted,
        )

    @mock.patch("services.comfy_lora_resolver.requests.get")
    def test_inventory_fetch_reads_lora_loader_api(self, get):
        from services.comfy_lora_resolver import clear_comfy_lora_inventory_cache, fetch_comfy_lora_inventory

        clear_comfy_lora_inventory_cache()
        response = mock.Mock()
        response.json.return_value = {
            "LoraLoader": {
                "input": {"required": {"lora_name": [[r"folder\style.safetensors"]]}}
            }
        }
        response.raise_for_status.return_value = None
        get.return_value = response
        self.assertEqual(fetch_comfy_lora_inventory(refresh=True).candidates, (r"folder\style.safetensors",))
        get.assert_called_once()


if __name__ == "__main__":
    unittest.main()

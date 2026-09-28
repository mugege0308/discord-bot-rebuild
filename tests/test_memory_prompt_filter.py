import unittest
from unittest.mock import patch

from services.memory import build_memory_summary


class PromptMemoryCategoryFilterTests(unittest.TestCase):
    _MEMORIES = [
        {"id": 2, "text": "プロンプトは可愛い系、暖色、夕日、神社、和服が好き"},
        {"id": 3, "text": "寒色が好き"},
        {"id": 4, "text": "和風な服が好き"},
    ]

    def _summary(self, query: str) -> str:
        with patch("services.memory.search_memories", return_value=self._MEMORIES):
            return build_memory_summary(1, query, mode="prompt", compact=True)

    def test_character_pose_camera_request_does_not_inject_visual_preferences(self):
        summary = self._summary("原神の胡桃が腕を組んで少し下から")

        self.assertNotIn("Memory tags", summary)
        self.assertNotIn("cool colors", summary)
        self.assertNotIn("warm lighting", summary)
        self.assertNotIn("cute", summary)

    def test_explicit_cool_color_request_allows_matching_memory_tags_only(self):
        summary = self._summary("原神の胡桃を寒色系で腕を組んで少し下から")

        self.assertIn("cool colors", summary)
        self.assertIn("blue tones", summary)
        self.assertNotIn("orange tones", summary)
        self.assertNotIn("warm lighting", summary)
        self.assertNotIn("cute", summary)

    def test_explicit_usual_palette_allows_color_memory(self):
        summary = self._summary("原神の胡桃をいつもの色味で腕を組んで")

        self.assertIn("cool colors", summary)
        self.assertIn("blue tones", summary)
        self.assertIn("orange tones", summary)
        self.assertNotIn("cute", summary)


if __name__ == "__main__":
    unittest.main()

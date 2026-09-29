import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services.generation_history import (
    SCHEMA_VERSION,
    GenerationRequestRecord,
    get_generation_request,
    init_db,
    save_generation_history,
)
from services.sd_control import GeneratedImage


class GenerationHistoryCompositionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp_dir.cleanup)
        self.db_file = Path(self.temp_dir.name) / "generation_history.db"
        self.data_patcher = mock.patch("services.generation_history.DATA_DIR", Path(self.temp_dir.name))
        self.db_patcher = mock.patch("services.generation_history.DB_FILE", self.db_file)
        self.data_patcher.start()
        self.db_patcher.start()
        self.addCleanup(self.data_patcher.stop)
        self.addCleanup(self.db_patcher.stop)

    def _request(self, composition_preset="none"):
        return GenerationRequestRecord(
            user_id=1,
            request_text="テスト",
            reference_id=None,
            positive="1girl, full body",
            negative="low quality",
            summary="テスト",
            assumptions=(),
            workflow_file=None,
            workflow_hash=None,
            selection_used=False,
            selection_message="",
            composition_preset=composition_preset,
        )

    def test_fresh_schema_records_composition(self):
        init_db()
        request_id = save_generation_history(
            self._request("full_body"),
            [GeneratedImage(path=Path("image.png"), batch_index=1)],
        )
        self.assertIsNotNone(request_id)
        detail = get_generation_request(request_id)
        self.assertIsNotNone(detail)
        self.assertEqual("full_body", detail.composition_preset)
        with sqlite3.connect(self.db_file) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(generation_requests)")
            }
        self.assertEqual(SCHEMA_VERSION, version)
        self.assertIn("composition_preset", columns)

    def test_v5_database_gets_additive_composition_column(self):
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_file) as connection:
            connection.execute(
                """
                CREATE TABLE generation_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    request_text TEXT NOT NULL,
                    reference_id INTEGER,
                    model_key TEXT,
                    positive TEXT NOT NULL,
                    negative TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    assumptions_json TEXT NOT NULL,
                    workflow_file TEXT,
                    workflow_hash TEXT,
                    selection_used INTEGER NOT NULL DEFAULT 0,
                    selection_message TEXT,
                    parent_request_id INTEGER,
                    regeneration_type TEXT,
                    custom_instruction TEXT
                )
                """
            )
            connection.execute("PRAGMA user_version = 5")
            connection.commit()
        init_db()
        with sqlite3.connect(self.db_file) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(generation_requests)")
            }
        self.assertEqual(SCHEMA_VERSION, version)
        self.assertIn("composition_preset", columns)


if __name__ == "__main__":
    unittest.main()

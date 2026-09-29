from __future__ import annotations

import io
import os
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from services import lora_upload


def image_bytes(*, color: tuple[int, int, int] = (120, 160, 220), fmt: str = "PNG") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(output, format=fmt)
    return output.getvalue()


class LoraUploadServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.environment = patch.dict(
            os.environ,
            {
                "LORA_UPLOAD_ROOT": str(self.root / "sessions"),
                "LORA_UPLOAD_MAX_ATTACHMENT_BYTES": "10485760",
                "LORA_UPLOAD_MAX_ZIP_BYTES": "10485760",
            },
            clear=False,
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temp.cleanup()

    def new_session(self) -> dict:
        return lora_upload.create_session(
            owner_id=1,
            guild_id=2,
            channel_id=3,
            message_id=None,
            dataset_name="スマホテスト",
            lora_type="additional",
            trigger_tag="test trigger",
            recommended_strength=0.8,
        )

    def test_direct_image_and_nested_zip_caption_are_registered(self) -> None:
        session = self.new_session()
        direct = lora_upload.ingest_attachment(
            session["session_id"], filename="phone.PNG", data=image_bytes()
        )
        self.assertEqual(direct["width"], 32)
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr("unnecessary-parent/sample.webp", image_bytes(fmt="WEBP"))
            archive.writestr("unnecessary-parent/sample.txt", "additional tag, blue eyes\n")
            archive.writestr("__MACOSX/ignored.txt", "ignored")
        result = lora_upload.ingest_attachment(
            session["session_id"], filename="images.zip", data=archive_buffer.getvalue()
        )
        self.assertEqual(result["image_count"], 1)
        saved = lora_upload.get_session(session["session_id"])
        self.assertEqual(lora_upload.summary(saved)["image_count"], 2)
        self.assertEqual(lora_upload.summary(saved)["caption_count"], 1)
        _session, staging = lora_upload.finalize_to_staging(session["session_id"])
        self.assertEqual(
            len(list(staging.rglob("*.png"))) + len(list(staging.rglob("*.webp"))),
            2,
        )
        self.assertTrue((staging / "unnecessary-parent" / "sample.webp").is_file())
        self.assertTrue(any(path.suffix == ".txt" for path in staging.rglob("*.txt")))

    def test_sixteen_image_caption_pairs_keep_legacy_manifest_without_training_design(self) -> None:
        session = self.new_session()
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            for index in range(16):
                image_name = f"dataset/{index + 1:02d}.png"
                caption_name = f"dataset/{index + 1:02d}.txt"
                archive.writestr(
                    image_name,
                    image_bytes(color=((index * 17) % 256, 80, 160)),
                )
                archive.writestr(caption_name, f"test_trigger, sample_{index + 1}\n")
            archive.writestr("lora_manifest.json", '{"images": 16, "captions": 16}')

        result = lora_upload.ingest_attachment(
            session["session_id"], filename="dataset.zip", data=archive_buffer.getvalue()
        )
        self.assertEqual(result["image_count"], 16)
        saved = lora_upload.get_session(session["session_id"])
        summary = lora_upload.summary(saved)
        self.assertEqual(summary["image_count"], 16)
        self.assertEqual(summary["caption_count"], 16)
        self.assertEqual(len(saved["items"]), 16)
        self.assertTrue(all(lora_upload.is_item_included(item) for item in saved["items"]))
        self.assertNotIn("lora_manifest.json", {item["original_name"] for item in saved["items"]})

        _session, staging = lora_upload.finalize_to_staging(session["session_id"])
        self.assertEqual(len(list(staging.rglob("*.png"))), 16)
        self.assertEqual(len(list(staging.rglob("*.txt"))), 16)
        self.assertTrue((staging / "training" / "lora_manifest.json").is_file())

    def test_weighted_zip_keeps_subset_paths_and_embedded_training_files(self) -> None:
        session = self.new_session()
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            for subset, count in (("identity_detail", 4), ("general", 8)):
                for index in range(count):
                    image_name = f"images/{subset}/{index + 1:02d}.png"
                    archive.writestr(image_name, image_bytes(color=(index + (0 if subset == "general" else 40), 80, 160)))
                    archive.writestr(
                        f"images/{subset}/{index + 1:02d}.txt",
                        f"test_trigger, {subset}_{index + 1}\n",
                    )
            archive.writestr(
                "training/dataset.toml",
                """[[datasets]]
  [[datasets.subsets]]
  image_dir = "../images/general"
  num_repeats = 8

  [[datasets.subsets]]
  image_dir = "../images/identity_detail"
  num_repeats = 24
""",
            )
            archive.writestr("training/config.toml", "max_train_epochs = 6\n")
            archive.writestr("lora_manifest.json", '{"recommended_strength": 0.7}')

        result = lora_upload.ingest_attachment(
            session["session_id"],
            filename="sutera_starrail_lora_dataset_v7_weighted.zip",
            data=archive_buffer.getvalue(),
        )
        self.assertEqual(result["image_count"], 12)
        saved = lora_upload.get_session(session["session_id"])
        self.assertEqual(
            {item["original_name"] for item in saved["items"]},
            {
                *{f"images/identity_detail/{index:02d}.png" for index in range(1, 5)},
                *{f"images/general/{index:02d}.png" for index in range(1, 9)},
            },
        )
        self.assertEqual(
            set(saved["embedded_training"]["files"]),
            {"dataset.toml", "config.toml", "lora_manifest.json"},
        )
        _session, staging = lora_upload.finalize_to_staging(session["session_id"])
        self.assertTrue((staging / "images" / "general" / "01.png").is_file())
        self.assertTrue((staging / "images" / "identity_detail" / "01.png").is_file())
        self.assertTrue((staging / "training" / "dataset.toml").is_file())

    def test_inclusion_rule_accepts_legacy_fields_consistently(self) -> None:
        session = {
            "items": [
                {"id": 1, "included": True},
                {"id": 2, "include": True},
                {"id": 3, "accepted": True},
                {"id": 4, "excluded": False},
                {"id": 5, "included": "false"},
                {"id": 6, "accepted": False},
                {"id": 7, "excluded": True},
            ]
        }
        self.assertEqual([item["id"] for item in lora_upload.included_items(session)], [1, 2, 3, 4])

    def test_new_session_replaces_only_open_sessions_with_backup(self) -> None:
        old = self.new_session()
        lora_upload.ingest_attachment(old["session_id"], filename="old.png", data=image_bytes())
        other_owner = lora_upload.create_session(
            owner_id=2,
            guild_id=2,
            channel_id=3,
            message_id=None,
            dataset_name="別ユーザー",
            lora_type="style",
            trigger_tag="",
            recommended_strength=0.5,
        )

        replacement = self.new_session()
        replaced = lora_upload.get_session(old["session_id"])
        self.assertEqual(replaced["status"], "cancelled")
        self.assertEqual(replaced["cancel_reason"], "replaced_by_new_session")
        backup = Path(replaced["backup_root"])
        self.assertTrue((backup / "session.json").is_file())
        self.assertTrue((backup / "session_root" / "files" / "item_0001.png").is_file())
        self.assertTrue((Path(old["root"]) / "files" / "item_0001.png").is_file())
        self.assertEqual(lora_upload.get_session(other_owner["session_id"])["status"], "open")
        self.assertEqual(lora_upload.list_sessions(1)[0]["session_id"], replacement["session_id"])

        lora_upload.mark_plan_created(
            replacement["session_id"],
            {"info_path": "info.json", "plan_dir": "plan", "output_name": "out"},
        )
        newest = self.new_session()
        self.assertEqual(lora_upload.get_session(replacement["session_id"])["status"], "cancelled")
        self.assertEqual(
            lora_upload.get_session(replacement["session_id"])["cancel_reason"],
            "replaced_by_new_session",
        )
        self.assertEqual(lora_upload.get_session(newest["session_id"])["status"], "open")

    def test_concurrent_creation_leaves_one_open_session_per_owner(self) -> None:
        def create(_: int) -> dict:
            return self.new_session()

        with ThreadPoolExecutor(max_workers=4) as executor:
            sessions = list(executor.map(create, range(4)))
        open_sessions = lora_upload.list_sessions(1)
        self.assertEqual(len(open_sessions), 1)
        open_id = open_sessions[0]["session_id"]
        self.assertIn(open_id, {session["session_id"] for session in sessions})
        for session in sessions:
            if session["session_id"] == open_id:
                continue
            saved = lora_upload.get_session(session["session_id"])
            self.assertEqual(saved["status"], "cancelled")
            self.assertEqual(saved["cancel_reason"], "replaced_by_new_session")

    def test_new_session_does_not_replace_session_owned_by_running_job(self) -> None:
        old = self.new_session()
        jobs = self.root / "jobs"
        jobs.mkdir()
        (jobs / "running.json").write_text(
            '{"session_id": "' + old["session_id"] + '", "status": "running"}',
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"LORA_JOB_ROOT": str(jobs)}, clear=False):
            replacement = self.new_session()
        self.assertEqual(lora_upload.get_session(old["session_id"])["status"], "open")
        self.assertEqual(lora_upload.get_session(replacement["session_id"])["status"], "open")

    def test_duplicate_and_unsupported_files_are_reported(self) -> None:
        session = self.new_session()
        data = image_bytes()
        first = lora_upload.ingest_attachment(session["session_id"], filename="a.png", data=data)
        self.assertFalse(first.get("duplicate", False))
        duplicate = lora_upload.ingest_attachment(session["session_id"], filename="b.png", data=data)
        self.assertTrue(duplicate["duplicate"])
        with self.assertRaises(lora_upload.LoraUploadError):
            lora_upload.ingest_attachment(session["session_id"], filename="notes.pdf", data=b"pdf")
        saved = lora_upload.get_session(session["session_id"])
        self.assertEqual(len(saved["items"]), 1)
        self.assertEqual(len(saved["duplicates"]), 1)
        self.assertTrue(any(item["file"] == "notes.pdf" for item in saved["rejected"]))

    def test_empty_corrupt_and_zip_slip_are_rejected(self) -> None:
        for payload in (b"", b"not a zip"):
            session = self.new_session()
            with self.assertRaises(lora_upload.LoraUploadError):
                lora_upload.ingest_attachment(session["session_id"], filename="bad.zip", data=payload)
            lora_upload.cancel_session(session["session_id"])

        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr("../escape.png", image_bytes())
        session = self.new_session()
        with self.assertRaises(lora_upload.LoraUploadError):
            lora_upload.ingest_attachment(
                session["session_id"], filename="slip.zip", data=archive_buffer.getvalue()
            )
        self.assertFalse((self.root / "escape.png").exists())

        lora_upload.cancel_session(session["session_id"])
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            symlink = zipfile.ZipInfo("link.png")
            symlink.external_attr = 0o120777 << 16
            archive.writestr(symlink, "target.png")
        session = self.new_session()
        with self.assertRaises(lora_upload.LoraUploadError):
            lora_upload.ingest_attachment(
                session["session_id"], filename="symlink.zip", data=archive_buffer.getvalue()
            )

    def test_dataset_name_and_character_trigger_are_validated(self) -> None:
        with self.assertRaises(lora_upload.LoraUploadError):
            lora_upload.create_session(
                owner_id=1, guild_id=None, channel_id=1, message_id=None,
                dataset_name="..\\escape", lora_type="style", trigger_tag="", recommended_strength=0.5,
            )
        with self.assertRaises(lora_upload.LoraUploadError):
            lora_upload.create_session(
                owner_id=1, guild_id=None, channel_id=1, message_id=None,
                dataset_name="character", lora_type="character", trigger_tag="", recommended_strength=0.5,
            )

    def test_cancel_removes_only_session_area(self) -> None:
        session = self.new_session()
        lora_upload.ingest_attachment(session["session_id"], filename="a.png", data=image_bytes())
        outside = self.root / "keep.txt"
        outside.write_text("keep", encoding="utf-8")
        lora_upload.cancel_session(session["session_id"])
        self.assertTrue(outside.exists())
        self.assertFalse((self.root / "sessions" / session["session_id"]).exists())


if __name__ == "__main__":
    unittest.main()

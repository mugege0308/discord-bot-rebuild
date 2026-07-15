import unittest
from pathlib import Path

from services.dev_worktree_parser import parse_worktree_list_porcelain_z


def _build_porcelain_z(lines: list[str]) -> bytes:
    """Build the NUL-delimited byte stream `git worktree list --porcelain
    -z` would produce for the given sequence of "lines" (using "" for what
    would be a blank line / record separator in the non -z format).
    """
    return b"\0".join(line.encode("utf-8") for line in lines) + b"\0"


class DevWorktreeParserTestCase(unittest.TestCase):
    def test_parse_empty_output(self):
        self.assertEqual(parse_worktree_list_porcelain_z(b""), [])

    def test_parse_multiple_entries(self):
        raw = _build_porcelain_z(
            [
                "worktree /repo",
                "HEAD abc123",
                "branch refs/heads/main",
                "",
                "worktree /repo/work/task-a",
                "HEAD def456",
                "branch refs/heads/dev-task/aaaaaaaa",
                "",
                "worktree /repo/work/task-b",
                "HEAD 789abc",
                "detached",
                "",
                "worktree /repo/bare",
                "bare",
                "",
                "worktree /repo/work/task-c",
                "HEAD 111222",
                "branch refs/heads/dev-task/cccccccc",
                "locked stale reason",
                "prunable gitdir file points to non-existent location",
            ]
        )
        entries = parse_worktree_list_porcelain_z(raw)
        self.assertEqual(len(entries), 5)

        self.assertEqual(entries[0].path, Path("/repo"))
        self.assertEqual(entries[0].branch, "refs/heads/main")
        self.assertFalse(entries[0].detached)

        self.assertEqual(entries[1].path, Path("/repo/work/task-a"))
        self.assertEqual(entries[1].branch, "refs/heads/dev-task/aaaaaaaa")

        self.assertEqual(entries[2].path, Path("/repo/work/task-b"))
        self.assertTrue(entries[2].detached)
        self.assertIsNone(entries[2].branch)

        self.assertTrue(entries[3].bare)

        self.assertEqual(entries[4].path, Path("/repo/work/task-c"))
        self.assertTrue(entries[4].locked)
        self.assertTrue(entries[4].prunable)

    def test_parse_rejects_field_before_any_worktree_header(self):
        raw = _build_porcelain_z(["branch refs/heads/main", ""])
        self.assertIsNone(parse_worktree_list_porcelain_z(raw))

    def test_parse_rejects_completely_unstructured_garbage(self):
        self.assertIsNone(parse_worktree_list_porcelain_z(b"not porcelain output at all"))

    def test_parse_rejects_input_missing_trailing_nul(self):
        # Well-formed -z output always ends in NUL; a stream that stops
        # mid-field (as a byte-capped truncation would) must never be
        # parsed as if it were a complete record.
        raw = b"worktree /repo\0HEAD abc123\0branch refs/heads/main"
        self.assertIsNone(parse_worktree_list_porcelain_z(raw))

    def test_parse_rejects_multi_record_input_missing_trailing_nul(self):
        raw = (
            b"worktree /repo\0HEAD abc123\0branch refs/heads/main\0\0"
            b"worktree /repo/work/task-a\0HEAD def456\0branch refs/heads/dev-task/xyz"
        )
        self.assertIsNone(parse_worktree_list_porcelain_z(raw))

    def test_parse_ignores_unknown_fields_in_an_otherwise_valid_record(self):
        raw = _build_porcelain_z(["worktree /repo", "some-future-field value", "branch refs/heads/main", ""])
        entries = parse_worktree_list_porcelain_z(raw)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].branch, "refs/heads/main")

    def test_parse_survives_path_with_space_japanese_drive_letter_backslash_and_embedded_newline(self):
        tricky_path = "C:\\Users\\テスト ユーザー\\work\\abc\ndef ghi"
        raw = _build_porcelain_z(
            ["worktree " + tricky_path, "HEAD abc123", "branch refs/heads/dev-task/xyz", ""]
        )
        entries = parse_worktree_list_porcelain_z(raw)
        self.assertEqual(len(entries), 1)
        # The embedded newline must never be mistaken for a record
        # boundary: only NUL bytes separate fields/records in -z output, so
        # the full path - including the newline - round-trips intact.
        self.assertEqual(entries[0].path, Path(tricky_path))

    def test_parse_does_not_depend_on_quote_settings(self):
        # -z output is never quoted (unlike the non -z porcelain format,
        # which is sensitive to core.quotePath); a raw non-ASCII path must
        # come through unescaped and unmodified.
        raw = _build_porcelain_z(["worktree /repo/work/日本語パス", "HEAD abc123", "detached", ""])
        entries = parse_worktree_list_porcelain_z(raw)
        self.assertEqual(entries[0].path, Path("/repo/work/日本語パス"))


if __name__ == "__main__":
    unittest.main()

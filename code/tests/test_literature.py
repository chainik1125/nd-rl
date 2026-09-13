"""Safety and asynchronous import checks for the local literature command."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "literature", Path(__file__).resolve().parents[1] / "tools" / "literature.py"
)
assert SPEC and SPEC.loader
literature = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(literature)

BIB = "@misc{example, title={A paper}, year={2026}}"
CONFIG = {"libraryID": 1, "id": 36, "key": "PROJECT"}


class LiteratureTests(unittest.TestCase):
    def test_rejects_multi_entry_import(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            literature.parse_one(BIB + "\n@misc{other, title={Another paper}}")

    def test_replaces_citekey_without_rewriting_entry(self):
        entry = literature.parse_one(BIB)
        updated = literature.replace_citekey(BIB, entry, "nd2026")
        self.assertIn("@misc{nd2026,", updated)
        self.assertIn("title={A paper}", updated)
        self.assertEqual(entry["ID"], "nd2026")

    def test_rejects_markdown_citation_as_citekey(self):
        with self.assertRaisesRegex(ValueError, "without @ or brackets"):
            literature.validate_citekey("[@nd2026]")

    def test_rejects_empty_citekey(self):
        with self.assertRaisesRegex(ValueError, "must start"):
            literature.validate_citekey("")

    @patch.object(literature, "fetch_bib", return_value=BIB)
    @patch.object(literature, "matches", return_value=[{"citekey": "example"}])
    @patch.object(literature, "rpc", return_value={"example": [{"key": "PROJECT"}]})
    @patch.object(literature, "wait_for_export")
    @patch.object(literature, "post")
    def test_existing_paper_causes_no_write(self, post, wait, *_):
        literature.add("example.bib", CONFIG)
        post.assert_not_called()
        wait.assert_called_once_with("example")

    @patch.object(literature, "fetch_bib", return_value=BIB)
    @patch.object(literature, "matches", return_value=[{"citekey": "example"}])
    @patch.object(literature, "post")
    def test_citekey_does_not_rename_existing_paper(self, post, *_):
        with self.assertRaisesRegex(ValueError, "only names new imports"):
            literature.add("example.bib", CONFIG, "nd2026")
        post.assert_not_called()

    @patch.object(literature, "fetch_bib", return_value=BIB)
    @patch.object(literature, "matches", return_value=[])
    @patch.object(literature, "find_by_citekey", return_value=[{"citekey": "nd2026"}])
    @patch.object(literature, "post")
    def test_citekey_collision_prevents_import(self, post, *_):
        with self.assertRaisesRegex(ValueError, "already used"):
            literature.add("example.bib", CONFIG, "nd2026")
        post.assert_not_called()

    @patch.object(literature, "fetch_bib", return_value=BIB)
    @patch.object(literature, "matches", return_value=[])
    @patch.object(literature, "post", return_value={"libraryID": 2, "id": 36})
    def test_wrong_library_prevents_import(self, post, *_):
        with self.assertRaisesRegex(ValueError, "Nothing was imported"):
            literature.add("example.bib", CONFIG)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.args[0], "/connector/getSelectedCollection")

    @patch.object(literature, "fetch_bib", return_value=BIB)
    @patch.object(literature, "matches", return_value=[{"citekey": "example"}])
    @patch.object(literature, "rpc", return_value={"example": [{"key": "OTHER"}]})
    @patch.object(literature, "post")
    def test_existing_paper_elsewhere_does_not_duplicate(self, post, *_):
        with self.assertRaisesRegex(ValueError, "duplicate was not created"):
            literature.add("example.bib", CONFIG)
        post.assert_not_called()

    @patch.object(literature.time, "sleep")
    @patch.object(
        literature, "matches", side_effect=[[], [{"citekey": ""}], [{"citekey": "ready"}]]
    )
    def test_waits_for_delayed_citation_key(self, matches, _):
        self.assertEqual(literature.wait_for_record({"title": "A paper"}, CONFIG), "ready")
        self.assertEqual(matches.call_count, 3)

    @patch.object(literature.time, "sleep")
    @patch.object(
        literature,
        "find_by_citekey",
        side_effect=[[], [{"citekey": "nd2026"}]],
    )
    def test_waits_for_requested_citation_key(self, find, _):
        self.assertEqual(
            literature.wait_for_record({"title": "A paper"}, CONFIG, "nd2026"), "nd2026"
        )
        self.assertEqual(find.call_count, 2)

    @patch.object(literature, "fetch_bib", return_value=BIB)
    @patch.object(literature, "matches", return_value=[])
    @patch.object(literature, "post", side_effect=[{"libraryID": 1, "id": 36}, TimeoutError()])
    @patch.object(literature, "wait_for_record", return_value="example")
    @patch.object(literature, "rpc", return_value={"example": [{"key": "PROJECT"}]})
    @patch.object(literature, "wait_for_export")
    def test_timeout_reconciles_without_repeating_import(self, wait, rpc, record, post, *_):
        literature.add("example.bib", CONFIG)
        self.assertEqual(post.call_count, 2)
        wait.assert_called_once_with("example")


if __name__ == "__main__":
    unittest.main()

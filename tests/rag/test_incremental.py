"""Re-ingesting only what changed.

The ingestor deduplicates within a run and has no notion of a previous one, so a
thousand files with one edit costs a thousand embeddings. The manifest is the
missing half.
"""

from __future__ import annotations

from neurosurfer.rag.incremental import IngestManifest


def _manifest(tmp_path):
    return IngestManifest.load(tmp_path / "manifest.json")


class TestDiff:
    def test_a_first_run_is_all_new(self, tmp_path):
        delta = _manifest(tmp_path).diff({"a.md": "one", "b.md": "two"})

        assert delta.new == ["a.md", "b.md"]
        assert delta.to_ingest == ["a.md", "b.md"]

    def test_unchanged_sources_are_not_re_ingested(self, tmp_path):
        m = _manifest(tmp_path)
        m.record("a.md", "one", ["a:0"])

        delta = m.diff({"a.md": "one"})

        assert delta.unchanged == ["a.md"]
        assert delta.to_ingest == []
        assert delta.is_empty

    def test_an_edited_source_is_changed(self, tmp_path):
        m = _manifest(tmp_path)
        m.record("a.md", "one", ["a:0"])

        delta = m.diff({"a.md": "one, edited"})

        assert delta.changed == ["a.md"]
        assert delta.to_ingest == ["a.md"]

    def test_a_deleted_source_is_reported(self, tmp_path):
        m = _manifest(tmp_path)
        m.record("gone.md", "text", ["gone:0"])

        delta = m.diff({})

        assert delta.removed == ["gone.md"]

    def test_the_mixed_case(self, tmp_path):
        m = _manifest(tmp_path)
        m.record("same.md", "s", ["same:0"])
        m.record("edit.md", "before", ["edit:0"])
        m.record("gone.md", "g", ["gone:0"])

        delta = m.diff({"same.md": "s", "edit.md": "after", "new.md": "n"})

        assert (delta.new, delta.changed, delta.unchanged, delta.removed) == (
            ["new.md"], ["edit.md"], ["same.md"], ["gone.md"]
        )
        assert str(delta) == "1 new, 1 changed, 1 unchanged, 1 removed"


class TestStaleChunks:
    def test_a_changed_source_yields_its_old_chunk_ids(self, tmp_path):
        """Without this the options are stale text in the index or clearing the
        whole collection — which is what the manifest exists to avoid."""
        m = _manifest(tmp_path)
        m.record("a.md", "before", ["a:0", "a:1", "a:2"])

        delta = m.diff({"a.md": "after"})

        assert m.stale_chunk_ids(delta) == ["a:0", "a:1", "a:2"]

    def test_a_removed_source_yields_its_chunk_ids_too(self, tmp_path):
        m = _manifest(tmp_path)
        m.record("gone.md", "g", ["gone:0"])

        assert m.stale_chunk_ids(m.diff({})) == ["gone:0"]

    def test_an_unchanged_source_yields_nothing(self, tmp_path):
        m = _manifest(tmp_path)
        m.record("a.md", "same", ["a:0"])

        assert m.stale_chunk_ids(m.diff({"a.md": "same"})) == []


class TestPersistence:
    def test_it_round_trips(self, tmp_path):
        m = _manifest(tmp_path)
        m.record("a.md", "text", ["a:0", "a:1"])
        m.save()

        reloaded = _manifest(tmp_path)

        assert reloaded.diff({"a.md": "text"}).unchanged == ["a.md"]
        assert reloaded.chunks["a.md"] == ["a:0", "a:1"]

    def test_a_missing_manifest_is_an_empty_one(self, tmp_path):
        assert _manifest(tmp_path).sources == {}

    def test_a_corrupt_manifest_costs_a_re_ingest_not_a_failed_run(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text("{not json at all")

        m = IngestManifest.load(path)

        assert m.sources == {}
        assert m.diff({"a.md": "x"}).new == ["a.md"]

    def test_forgetting_removes_both_halves(self, tmp_path):
        m = _manifest(tmp_path)
        m.record("a.md", "x", ["a:0"])

        m.forget(["a.md"])

        assert m.sources == {} and m.chunks == {}

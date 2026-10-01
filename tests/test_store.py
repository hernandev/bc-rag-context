from types import SimpleNamespace

from bc_rag.chunking import Chunk
from bc_rag.embeddings import SparseVec
from bc_rag.store import HybridStore, facet_filter


def _keys(conditions) -> list[tuple[str, list[str]]]:
    return [(condition.key, list(condition.match.any)) for condition in conditions or []]


def test_every_key_must_match_and_values_are_any_of() -> None:
    built = facet_filter({"vendor": ["olo"], "area": ["engine", "admin"]})
    assert _keys(built.must) == [
        ("facets.vendor", ["olo"]),
        ("facets.area", ["engine", "admin"]),
    ]
    assert built.must_not is None


def test_exclude_becomes_must_not() -> None:
    built = facet_filter({"scope": ["internal"]}, {"vendor": ["olo", "yext"]})
    assert _keys(built.must) == [("facets.scope", ["internal"])]
    assert _keys(built.must_not) == [("facets.vendor", ["olo", "yext"])]


def test_exclude_alone() -> None:
    built = facet_filter(None, {"group": ["docs-loose"]})
    assert built.must is None
    assert _keys(built.must_not) == [("facets.group", ["docs-loose"])]


def test_no_facets_no_filter() -> None:
    assert facet_filter(None) is None
    assert facet_filter({}, {}) is None


class RecordingClient:
    def __init__(self, schema: dict | None = None) -> None:
        self.indexes: list[str] = []
        self.upserts: list = []
        self.schema = schema or {}

    def collection_exists(self, name: str) -> bool:
        return True

    def get_collection(self, name: str):
        return SimpleNamespace(payload_schema=self.schema)

    def create_payload_index(self, *, collection_name: str, field_name: str, field_schema) -> None:
        self.indexes.append(field_name)
        self.schema[field_name] = SimpleNamespace(points=0)

    def upsert(self, *, collection_name: str, points) -> None:
        self.upserts.extend(points)


def _store(client: RecordingClient, *, read_only: bool = False) -> HybridStore:
    store = HybridStore.__new__(HybridStore)
    store.client = client
    store.collection = "c"
    store.read_only = read_only
    store.url = None
    store.path = None
    store._indexed = set()
    return store


def _chunk(path: str, facets: dict) -> Chunk:
    return Chunk(
        path=path,
        language="markdown",
        kind="prose",
        symbol=None,
        heading_path=None,
        start_line=1,
        end_line=1,
        start_byte=0,
        end_byte=1,
        text="x",
        facets=facets,
    )


def test_new_keys_get_an_index_on_upsert_once() -> None:
    client = RecordingClient()
    store = _store(client)
    chunks = [_chunk("a.md", {"group": ["docs"], "area": ["engine"]})]
    sparse = [SparseVec(indices=[1], values=[1.0])]
    store.upsert_chunks(chunks, [[0.1]], sparse)
    store.upsert_chunks([_chunk("b.md", {"area": ["admin"], "provider": ["olo"]})], [[0.1]], sparse)
    assert client.indexes == ["facets.area", "facets.group", "facets.provider"]
    payload = client.upserts[0].payload
    assert payload["facets"] == {"group": ["docs"], "area": ["engine"]}
    assert "tags" not in payload
    assert "group" not in payload


def test_read_only_store_never_writes_indexes() -> None:
    client = RecordingClient()
    _store(client, read_only=True).ensure_payload_indexes()
    assert client.indexes == []


def test_base_index_is_created_once_and_existing_indexes_are_reused() -> None:
    client = RecordingClient(schema={"facets.area": SimpleNamespace(points=3)})
    store = _store(client)
    store.ensure_payload_indexes()
    store.ensure_payload_indexes()
    store.upsert_chunks(
        [_chunk("a.md", {"area": ["engine"]})], [[0.1]], [SparseVec(indices=[1], values=[1.0])]
    )
    assert client.indexes == ["path"]


def test_facet_keys_come_from_the_payload_schema() -> None:
    client = RecordingClient(
        schema={
            "path": SimpleNamespace(points=9),
            "facets.area": SimpleNamespace(points=3),
            "facets.empty": SimpleNamespace(points=0),
            "facets.group": SimpleNamespace(points=9),
        }
    )
    assert _store(client).facet_keys() == {"area": 3, "group": 9}

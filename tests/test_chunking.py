from pathlib import Path

from bc_rag.chunking import chunk_file


def test_typescript_keeps_a_function_whole() -> None:
    source = """
export function firstPresent(values: Array<string | null>): string | null {
  for (const value of values) {
    if (value) {
      return value;
    }
  }
  return null;
}

export class LocationProjectorService {
  firstPresent(values: Array<string | null>): string | null {
    return firstPresent(values);
  }
}
"""
    chunks = chunk_file(
        path=Path("src/LocationProjectorService.ts"),
        rel_path="src/LocationProjectorService.ts",
        language="typescript",
        text=source,
        max_chars=800,
        min_chars=20,
    )

    texts = "\n".join(chunk.text for chunk in chunks)
    assert "function firstPresent" in texts
    assert any(chunk.symbol == "firstPresent" or chunk.symbol == "LocationProjectorService" for chunk in chunks)


def test_package_json_renders_name_and_dependencies() -> None:
    text = """
{
  "name": "@bigcolony/engine-flows-locations",
  "version": "1.0.0",
  "dependencies": { "@bigcolony/engine-core-contracts": "workspace:*" },
  "scripts": { "test": "jest" }
}
"""
    chunks = chunk_file(
        path=Path("libs/engine/flows/engine-flows-locations/package.json"),
        rel_path="libs/engine/flows/engine-flows-locations/package.json",
        language="json",
        text=text,
        max_chars=4000,
        min_chars=10,
    )
    assert chunks
    assert chunks[0].kind == "package-json"
    assert "@bigcolony/engine-flows-locations" in chunks[0].text
    assert "engine-core-contracts" in chunks[0].text


def test_markdown_splits_on_headings() -> None:
    source = """# Locations

Intro.

## Overrides write versus read

Store writes the full object.

## Projection

Null is absent.
"""
    chunks = chunk_file(
        path=Path("docs/locations.md"),
        rel_path="docs/locations.md",
        language="markdown",
        text=source,
        max_chars=4000,
        min_chars=10,
    )

    headings = [chunk.heading_path for chunk in chunks if chunk.heading_path]
    assert any(heading and "Overrides" in heading for heading in headings)
    assert any(heading and "Projection" in heading for heading in headings)

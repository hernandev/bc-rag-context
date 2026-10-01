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
    assert any(
        chunk.symbol == "firstPresent" or chunk.symbol == "LocationProjectorService"
        for chunk in chunks
    )


def test_comment_above_a_declaration_stays_with_that_chunk() -> None:
    source = """
import { EngineScope } from '@bigcolony/engine-core-models';

/**
 * Default cache TTL for local favorites (10 minutes).
 */
const LOCAL_FAVORITES_CACHE_TTL_SECONDS = 600;

/**
 * Input for the action.
 */
export class LocationFlowGetLocalFavoritesActionInput {
  /**
   * Internal location identifier.
   */
  locationId!: string;
}
"""
    chunks = chunk_file(
        path=Path("src/Action.ts"),
        rel_path="src/Action.ts",
        language="typescript",
        text=source,
        max_chars=8000,
        min_chars=20,
    )
    by_symbol = {chunk.symbol: chunk.text for chunk in chunks}
    assert "Default cache TTL" in by_symbol["LOCAL_FAVORITES_CACHE_TTL_SECONDS"]
    assert "Input for the action." in by_symbol["LocationFlowGetLocalFavoritesActionInput"]
    assert "Internal location identifier." in by_symbol["LocationFlowGetLocalFavoritesActionInput"]


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


def test_markdown_does_not_split_on_headings_inside_code() -> None:
    source = """# Doc

```markdown
## This is an example heading

    # indented heading in a fence
```

    # four spaces, still code

Real text.
"""
    chunks = chunk_file(
        path=Path("docs/sample.md"),
        rel_path="docs/sample.md",
        language="markdown",
        text=source,
        max_chars=8000,
        min_chars=1,
    )
    assert len(chunks) == 1
    assert "This is an example heading" in chunks[0].text
    assert "four spaces, still code" in chunks[0].text


def test_markdown_splits_transcripts_on_rules_and_details() -> None:
    source = """# Chat

---

<details>
<summary>2026-07-25T02:44:06.592Z - user: Not a new chunk</summary>

## Not a new chunk

```
# still inside the message
```

</details>

---

<details>
<summary>2026-07-25T02:44:10.000Z - agent: Second message</summary>

Second message.
</details>
"""
    chunks = chunk_file(
        path=Path("docs/chat.md"),
        rel_path="docs/chat.md",
        language="markdown",
        text=source,
        max_chars=8000,
        min_chars=1,
    )
    user = next(chunk for chunk in chunks if chunk.facets.get("role") == ["user"])
    assert "still inside the message" in user.text
    assert "Second message." not in user.text
    agent = next(chunk for chunk in chunks if chunk.facets.get("role") == ["agent"])
    assert agent.heading_path is not None
    assert agent.heading_path.startswith("Chat > ")
    assert "Not a new chunk" in agent.text
    assert agent.facets["role"] == ["agent"]
    assert agent.facets["day"] == ["2026-07-25"]
    assert agent.facets["at"][0].startswith("2026-07-25T")


def test_long_turn_repeats_summary_on_the_next_piece() -> None:
    body = "row\n" * 500
    source = f"""# Chat

<details>
<summary>2026-07-25T02:44:06.592Z - user: short question</summary>

```
please look
```

</details>

---

<details>
<summary>2026-07-25T02:44:06.592Z - agent: long answer</summary>

```
{body}
```

</details>
"""
    chunks = chunk_file(
        path=Path("docs/long.md"),
        rel_path="docs/long.md",
        language="markdown",
        text=source,
        max_chars=800,
        min_chars=1,
    )
    agent_parts = [chunk for chunk in chunks if chunk.facets.get("role") == ["agent"]]
    assert len(agent_parts) > 1
    assert all("long answer" in chunk.text for chunk in agent_parts)
    assert all(
        chunk.heading_path and chunk.heading_path.startswith("Chat > ")
        for chunk in agent_parts
    )
    assert "short question" in agent_parts[0].text
    assert "User: short question" in agent_parts[1].text


def test_markdown_splits_on_pandoc_plain_rule() -> None:
    rule = "-" * 72
    source = f"""# Chat

one message

{rule}

two message
"""
    chunks = chunk_file(
        path=Path("docs/plain.md"),
        rel_path="docs/plain.md",
        language="markdown",
        text=source,
        max_chars=8000,
        min_chars=1,
    )
    texts = [chunk.text for chunk in chunks]
    assert any("one message" in text and "two message" not in text for text in texts)
    assert any("two message" in text and "one message" not in text for text in texts)
    assert not any(rule in text for text in texts)

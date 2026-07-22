"""
Unit tests for collection_generator batching logic.
No Claude API calls are made — _raw_call is mocked throughout.
"""
import json
import pytest
from unittest.mock import patch, call

from agent.collection_generator import (
    _make_batches,
    _validate_batch,
    _parse_with_diagnostics,
    _is_truncated_error,
    _generate_batch,
    _TruncatedResponseError,
    _FULL_BLEED_TERMS,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _poster(index: int, **overrides) -> dict:
    """Minimal valid raw poster dict for testing."""
    base = {
        "index": index,
        "title": f"Unique Poster Title {index}",
        "subject": f"Unique subject matter for poster {index}",
        "scene_concept": f"Unique detailed scene concept for poster {index}",
        "storytelling_focus": f"Unique storytelling angle {index}",
        "unique_hook": f"Unique visual hook number {index}",
        "image_prompt": (
            f"Full bleed composition, edge-to-edge illustration, "
            f"poster {index} specific scene, ink linework, watercolour brushwork, "
            f"painterly illustration technique, gouache texture"
        ),
        "negative_prompt": "border, frame, watermark",
        "aspect_ratio": "2:3",
        "focal_point": "centre",
        "foreground_elements": ["foreground element"],
        "midground_elements": ["midground element"],
        "background_elements": ["background element"],
        "palette_variation": ["deep blue", "warm ochre"],
        "lighting_variation": "warm afternoon light",
        "weather_or_time_variation": "clear morning",
        "consistency_notes": [f"Poster {index} maintains collection style."],
        "suggested_etsy_title": f"Etsy Title for Poster {index}",
        "suggested_etsy_tags": [f"tag{i:02d}poster{index}" for i in range(1, 14)],
        "mockup_room_style": "minimalist Japandi",
    }
    base.update(overrides)
    return base


def _batch_response(indexes: list[int], **poster_overrides) -> str:
    """Build a valid JSON batch response for the given poster indexes."""
    return json.dumps({
        "poster_items": [_poster(i, **poster_overrides) for i in indexes]
    })


# ── Partition tests ────────────────────────────────────────────────────────────

def test_partition_size_3():
    assert _make_batches(3) == [[1, 2], [3]]

def test_partition_size_4():
    assert _make_batches(4) == [[1, 2], [3, 4]]

def test_partition_size_5():
    assert _make_batches(5) == [[1, 2], [3, 4], [5]]

def test_partition_size_8():
    assert _make_batches(8) == [[1, 2], [3, 4], [5, 6], [7, 8]]

def test_partition_custom_batch_size():
    assert _make_batches(5, batch_size=3) == [[1, 2, 3], [4, 5]]


# ── Batch validation: index errors ────────────────────────────────────────────

def test_wrong_returned_indexes():
    items = [_poster(1), _poster(3)]  # returned [1,3] but requested [1,2]
    with pytest.raises(ValueError, match=r"\[1, 2\]"):
        _validate_batch(items, [1, 2], [], "posters")

def test_wrong_count():
    items = [_poster(1)]  # only 1 returned but 2 requested
    with pytest.raises(ValueError, match="expected 2"):
        _validate_batch(items, [1, 2], [], "posters")

def test_duplicate_indexes_within_batch():
    items = [_poster(1), _poster(1)]  # duplicate index
    with pytest.raises(ValueError, match="duplicate indexes"):
        _validate_batch(items, [1, 2], [], "posters")


# ── Batch validation: cross-batch duplicate checks ────────────────────────────

def test_duplicate_title_across_batches():
    prior = [_poster(1, title="Rainy Station Night")]
    current = [_poster(2, title="Rainy Station Night"), _poster(3)]
    with pytest.raises(ValueError, match="[Dd]uplicate title"):
        _validate_batch(current, [2, 3], prior, "posters")

def test_duplicate_subject_across_batches():
    prior = [_poster(1, subject="A lonely cat on a rainy windowsill")]
    current = [_poster(2, subject="A lonely cat on a rainy windowsill")]
    with pytest.raises(ValueError, match="[Dd]uplicate subject"):
        _validate_batch(current, [2], prior, "posters")

def test_duplicate_unique_hook_across_batches():
    prior = [_poster(1, unique_hook="Golden moonlight reflected in still water")]
    current = [_poster(2, unique_hook="Golden moonlight reflected in still water")]
    with pytest.raises(ValueError, match="[Dd]uplicate unique_hook"):
        _validate_batch(current, [2], prior, "posters")

def test_no_false_positive_on_fresh_batch():
    """First batch (no prior items) must pass without false duplicate errors."""
    items = [_poster(1), _poster(2)]
    _validate_batch(items, [1, 2], [], "posters")  # should not raise


# ── Batch validation: field-level checks ──────────────────────────────────────

def test_wrong_tag_count():
    items = [_poster(1, suggested_etsy_tags=["tag"] * 12)]  # 12 instead of 13
    with pytest.raises(ValueError, match="12 Etsy tags"):
        _validate_batch(items, [1], [], "posters")

def test_duplicate_etsy_tags_within_poster():
    tags = ["duplicate"] * 13
    items = [_poster(1, suggested_etsy_tags=tags)]
    with pytest.raises(ValueError, match="[Dd]uplicate Etsy tags"):
        _validate_batch(items, [1], [], "posters")

def test_missing_full_bleed():
    items = [_poster(1, image_prompt="A nice watercolour illustration with ink linework")]
    with pytest.raises(ValueError, match="full-bleed"):
        _validate_batch(items, [1], [], "posters")

def test_prohibited_paper_phrase():
    items = [_poster(1, image_prompt="Full bleed composition on cream paper texture")]
    with pytest.raises(ValueError, match="prohibited phrase"):
        _validate_batch(items, [1], [], "posters")

def test_empty_required_field():
    items = [_poster(1, title="")]
    with pytest.raises(ValueError, match="'title'"):
        _validate_batch(items, [1], [], "posters")


# ── Truncation detection ───────────────────────────────────────────────────────

def test_truncation_detected_unterminated_string():
    assert _is_truncated_error("unterminated string starting at line 10", "...") is True

def test_truncation_detected_unexpected_eof():
    assert _is_truncated_error("unexpected end of data", "...") is True

def test_truncation_detected_by_last_char():
    # Raw text ending mid-value (not }, ], or ")
    assert _is_truncated_error("some other error", '{"key": "val') is True

def test_no_truncation_on_valid_close():
    assert _is_truncated_error("some error", '{"key": "value"}') is False

def test_parse_diagnostics_non_truncation_raises_runtime():
    # Valid-looking JSON that is just structurally wrong (not truncated)
    raw = '{"key": invalid}'
    with pytest.raises(RuntimeError) as exc_info:
        _parse_with_diagnostics(raw, "test", [1])
    assert not isinstance(exc_info.value, _TruncatedResponseError)

def test_parse_diagnostics_truncation_raises_truncated():
    raw = '{"poster_items": [{"index": 1, "title": "unterminated'
    with pytest.raises(_TruncatedResponseError):
        _parse_with_diagnostics(raw, "test", [1])

def test_parse_diagnostics_includes_batch_indexes():
    raw = '{"poster_items": [{"index": 1, "title": "cut'
    with pytest.raises(_TruncatedResponseError, match=r"\[3, 4\]"):
        _parse_with_diagnostics(raw, "test", [3, 4])


# ── Retry behaviour ───────────────────────────────────────────────────────────

TRUNCATED_RAW = '{"poster_items": [{"index": 1, "title": "The Frog Wizard'

@patch("agent.collection_generator._raw_call")
def test_truncation_retry_succeeds(mock_raw):
    """First call truncates, second call succeeds — _raw_call called exactly twice."""
    valid = _batch_response([1])
    mock_raw.side_effect = [TRUNCATED_RAW, valid]

    import anthropic
    client = anthropic.Anthropic.__new__(anthropic.Anthropic)
    result = _generate_batch(client, "prompt", "posters", [1])

    assert result["poster_items"][0]["index"] == 1
    assert mock_raw.call_count == 2
    # Second call must have the compact suffix appended
    from agent.collection_generator import _COMPACT_SUFFIX
    assert _COMPACT_SUFFIX in mock_raw.call_args_list[1][0][1]

@patch("agent.collection_generator._raw_call")
def test_repeated_truncation_raises_diagnostic_error(mock_raw):
    """Both attempts truncate — _TruncatedResponseError raised with diagnostic info."""
    mock_raw.return_value = TRUNCATED_RAW

    import anthropic
    client = anthropic.Anthropic.__new__(anthropic.Anthropic)
    with pytest.raises(_TruncatedResponseError, match="likely_truncated: True"):
        _generate_batch(client, "prompt", "posters", [1])

    assert mock_raw.call_count == 2

@patch("agent.collection_generator._raw_call")
def test_non_truncation_error_does_not_retry(mock_raw):
    """A non-truncation JSON error must propagate immediately without retry."""
    mock_raw.return_value = '{"poster_items": invalid_json}'

    import anthropic
    client = anthropic.Anthropic.__new__(anthropic.Anthropic)
    with pytest.raises(RuntimeError) as exc_info:
        _generate_batch(client, "prompt", "posters", [1])

    # Must NOT be a truncation error, and must only have called once
    assert not isinstance(exc_info.value, _TruncatedResponseError)
    assert mock_raw.call_count == 1

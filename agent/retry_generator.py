import json
from dataclasses import dataclass, field
from agent.claude_client import ask
from agent.vision_critic import VisionReport

# Score threshold below which a fixable problem triggers retry
_RETRY_THRESHOLD = 5

REVISION_PROMPT = """\
You are an expert AI image prompt engineer for printable wall art.

An image was generated and reviewed. The review identified specific fixable problems.
Your job is to revise the image generation prompt to address ONLY those problems.

Poster concept:
{concept}

Current image generation prompt:
{image_prompt}

Current negative prompt:
{negative_prompt}

Vision Critic findings:
- Retry priority areas: {retry_priority}
- Weaknesses: {weaknesses}
- Improvement suggestions: {suggestions}

Revision rules:
- Address ONLY the identified retry priority areas and weaknesses.
- Preserve: core subject, niche identity, main colour palette (unless colour is a priority), \
intended customer, aspect ratio, set consistency rules, and all strengths.
- Do NOT introduce copyrighted characters, franchise names, artist names, or protected brand references.
- Do NOT overload the revised prompt with excessive new instructions.
- Do NOT create a completely different artwork.
- Only tighten the negative prompt for unwanted qualities explicitly identified in weaknesses.
- Do not add generic negatives unrelated to the report.
- Avoid contradictions between positive and negative prompts.

Return ONLY a valid JSON object — no markdown, no commentary:
{{
  "revised_image_prompt": "string",
  "revised_negative_prompt": "string",
  "changes_made": ["string — one change per item", "..."],
  "expected_improvements": ["string — realistic, not guaranteed", "..."],
  "preserved_elements": ["string — what was deliberately kept unchanged", "..."],
  "confidence_score": integer 1-10
}}
"""


@dataclass
class RetryPlan:
    should_retry: bool
    decision_reason: str
    retry_priority: list[str]
    revised_image_prompt: str | None
    revised_negative_prompt: str | None
    changes_made: list[str]
    expected_improvements: list[str]
    preserved_elements: list[str]
    confidence_score: int


def _should_retry(vr: VisionReport) -> tuple[bool, str]:
    """Decide whether to retry based on VisionReport. Returns (decision, reason)."""
    if vr.retry_recommended or vr.final_recommendation == "RETRY":
        return True, "Vision Critic recommends retry."

    low = []
    if vr.composition_score.score <= _RETRY_THRESHOLD:
        low.append(f"composition ({vr.composition_score.score}/10)")
    if vr.print_quality_score.score <= _RETRY_THRESHOLD:
        low.append(f"print quality ({vr.print_quality_score.score}/10)")
    if vr.collection_consistency_score.score <= _RETRY_THRESHOLD:
        low.append(f"collection consistency ({vr.collection_consistency_score.score}/10)")
    if vr.commercial_appeal_score.score <= _RETRY_THRESHOLD:
        low.append(f"commercial appeal ({vr.commercial_appeal_score.score}/10)")
    if low:
        return True, f"Critical score thresholds breached: {', '.join(low)}."

    return False, (
        f"Image meets quality thresholds "
        f"(overall {vr.overall_score.score}/10, recommendation: {vr.final_recommendation}). "
        "No retry warranted."
    )


def prepare_retry(
    poster_concept: dict,
    optimized_prompt: str,
    optimized_negative_prompt: str,
    vision_report: VisionReport,
    on_usage=None,
) -> RetryPlan:
    should_retry, decision_reason = _should_retry(vision_report)

    if not should_retry:
        return RetryPlan(
            should_retry=False,
            decision_reason=decision_reason,
            retry_priority=[],
            revised_image_prompt=None,
            revised_negative_prompt=None,
            changes_made=[],
            expected_improvements=[],
            preserved_elements=[],
            confidence_score=0,
        )

    prompt = REVISION_PROMPT.format(
        concept=json.dumps(poster_concept, indent=2),
        image_prompt=optimized_prompt,
        negative_prompt=optimized_negative_prompt,
        retry_priority=", ".join(vision_report.retry_priority) or "general quality",
        weaknesses="; ".join(vision_report.weaknesses) or "none specified",
        suggestions="; ".join(vision_report.improvement_suggestions) or "none specified",
    )

    raw = ask(prompt, on_usage=on_usage).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    d = json.loads(raw)

    return RetryPlan(
        should_retry=True,
        decision_reason=decision_reason,
        retry_priority=vision_report.retry_priority,
        revised_image_prompt=d["revised_image_prompt"],
        revised_negative_prompt=d["revised_negative_prompt"],
        changes_made=d.get("changes_made", []),
        expected_improvements=d.get("expected_improvements", []),
        preserved_elements=d.get("preserved_elements", []),
        confidence_score=d.get("confidence_score", 0),
    )

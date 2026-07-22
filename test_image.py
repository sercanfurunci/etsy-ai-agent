# ponytail: temporary test — remove once image generation is wired into main pipeline
import textwrap
from research.mock_provider import MockResearchProvider
from agent.analyzer import analyze
from agent.prompt_optimizer import optimize
from agent.vision_critic import review as vision_review, VisionReport
from agent.retry_generator import prepare_retry
from image.openai_provider import OpenAIImageProvider
from pathlib import Path

def _wrap(text: str, indent: int = 5) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=100, initial_indent=prefix, subsequent_indent=prefix)

if __name__ == "__main__":
    query = input("Görsel tipi / niş (örn. 'ukiyo-e cat poster'): ").strip()
    if not query:
        raise SystemExit("Query cannot be empty.")

    products = MockResearchProvider().search(query, limit=5)
    print(f"Analyzing {len(products)} mock products for: {query}\n")

    result = analyze([p.to_dict() for p in products], user_request=query, single_only=True)
    concepts = result.get("poster_concepts", [])

    print(f"Found {len(concepts)} poster concepts:\n")
    for i, c in enumerate(concepts, 1):
        print(f"  [{i}] {c.get('name', '—')}  [{c.get('single_or_set', '—')}]")
        print(f"      {c.get('niche', '—')} · {c.get('art_style', '—')}")

    print()
    raw = input(f"Select concept to generate [1–{len(concepts)}]: ").strip()
    try:
        choice = int(raw) - 1
        concept = concepts[choice]
    except (ValueError, IndexError):
        raise SystemExit("Invalid selection.")

    draft_prompt = concept.get("image_generation_prompt", "")
    draft_negative = concept.get("negative_prompt", "")
    print(f"\nSelected: {concept.get('name', '—')}")
    print(f"\nDraft image prompt:\n{_wrap(draft_prompt)}\n")

    print("Optimizing prompt with Claude...\n")
    try:
        opt = optimize(concept, draft_prompt, draft_negative)
    except Exception as e:
        raise SystemExit(f"[Optimizer error] {e}")

    report = opt["optimization_report"]
    print("Optimization report:")
    print(f"  Originality:           {report.get('originality_score')}/10")
    print(f"  Commercial appeal:     {report.get('commercial_appeal_score')}/10")
    print(f"  Print quality:         {report.get('print_quality_score')}/10")
    print(f"  Collection consistency:{report.get('collection_consistency_score')}/10")
    print(f"  Prompt clarity:        {report.get('prompt_clarity_score')}/10")
    print(f"  IP risk:               {report.get('ip_risk')}")
    print(f"\n  Changes made:")
    for change in report.get("changes_made", []):
        print(f"    • {change}")
    print(f"\n  Reasoning:\n{_wrap(report.get('reasoning', '—'))}\n")

    final_prompt = opt["optimized_image_prompt"]
    final_negative = opt["optimized_negative_prompt"]
    print(f"Optimized image prompt:\n{_wrap(final_prompt)}\n")
    print(f"Optimized negative prompt:\n{_wrap(final_negative)}\n")

    confirm = input("Generate image with optimized prompt? [y/N]: ").strip().lower()
    if confirm != "y":
        raise SystemExit("Aborted.")

    prompt = final_prompt
    print("\nGenerating...")
    try:
        provider = OpenAIImageProvider()
        path = provider.generate(prompt)
        print(f"\n[OK] Image saved: {path}")
    except ValueError as e:
        raise SystemExit(f"[Config error] {e}")
    except Exception as e:
        raise SystemExit(f"[API error] {e}")

    print("\nRunning Vision Critic...\n")
    try:
        vr = vision_review(concept, final_prompt, final_negative, path)
    except Exception as e:
        raise SystemExit(f"[Vision critic error] {e}")

    def _score(label: str, s) -> None:
        print(f"  {label}: {s.score}/10")
        print(f"    {_wrap(s.reason, indent=4)}")

    print("=" * 60)
    print("VISION CRITIC REPORT")
    print("=" * 60)

    print("\n  SCORES")
    print("  ------")
    _score("Overall (computed)      ", vr.overall_score)
    _score("Composition             ", vr.composition_score)
    _score("Originality             ", vr.originality_score)
    _score("Commercial appeal       ", vr.commercial_appeal_score)
    _score("Print quality           ", vr.print_quality_score)
    _score("Collection consistency  ", vr.collection_consistency_score)

    print("\n  MARKET")
    print("  ------")
    _score("Trend saturation        ", vr.trend_saturation_score)
    _score("Market uniqueness       ", vr.market_uniqueness_score)

    print(f"\n  IP similarity risk: {vr.ip_similarity_risk.upper()}")
    print(f"    {_wrap(vr.ip_similarity_reason, indent=4)}")

    print(f"\n  Strengths:")
    for s in vr.strengths:
        print(f"    + {s}")

    print(f"\n  Weaknesses:")
    for w in vr.weaknesses:
        print(f"    - {w}")

    if vr.improvement_suggestions:
        label = "Improvement suggestions" if not vr.retry_recommended else "Retry — priority improvements"
        print(f"\n  {label}:")
        for s in vr.improvement_suggestions:
            print(f"    • {s}")

    print(f"\n  Reasoning:\n{_wrap(vr.reasoning)}")

    print(f"\n  READINESS")
    print(f"  ---------")
    print(f"  Confidence:            {vr.confidence_score}/10")
    print(f"  Commercial readiness:  {vr.commercial_readiness}/10")
    print(f"  Print readiness:       {vr.print_readiness}/10")
    print(f"  Retry recommended:     {'YES' if vr.retry_recommended else 'no'}")
    if vr.retry_priority:
        print(f"  Retry priority:        {', '.join(vr.retry_priority)}")

    print(f"\n  FINAL RECOMMENDATION:  {vr.final_recommendation}")
    print("=" * 60)

    # ── Stage 13: Retry Generator ────────────────────────────────
    print("\nRunning Retry Generator...\n")
    try:
        plan = prepare_retry(concept, final_prompt, final_negative, vr)
    except Exception as e:
        raise SystemExit(f"[Retry generator error] {e}")

    print("─" * 60)
    print("RETRY PLAN")
    print("─" * 60)
    print(f"  Should retry:    {'YES' if plan.should_retry else 'no'}")
    print(f"  Decision reason: {plan.decision_reason}")

    if plan.should_retry:
        if plan.retry_priority:
            print(f"  Retry priority:  {', '.join(plan.retry_priority)}")
        print(f"  Confidence:      {plan.confidence_score}/10")
        print(f"\n  Changes to be made:")
        for c in plan.changes_made:
            print(f"    • {c}")
        print(f"\n  Expected improvements:")
        for e in plan.expected_improvements:
            print(f"    ~ {e}")
        print(f"\n  Preserved elements:")
        for p in plan.preserved_elements:
            print(f"    = {p}")
        print(f"\n  Revised image prompt:\n{_wrap(plan.revised_image_prompt or '')}")
        print(f"\n  Revised negative prompt:\n{_wrap(plan.revised_negative_prompt or '')}")
        print("─" * 60)

        confirm_retry = input("\nGenerate one revised image? [y/N]: ").strip().lower()
        if confirm_retry != "y":
            raise SystemExit("Retry skipped.")

        print("\nGenerating retry image...")
        try:
            retry_path_raw = Path(provider.generate(plan.revised_image_prompt))
            retry_path = retry_path_raw.with_stem(retry_path_raw.stem + "_retry")
            retry_path_raw.rename(retry_path)
            print(f"\n[OK] Retry image saved: {retry_path}")
        except Exception as e:
            raise SystemExit(f"[Retry generation error] {e}")

        print("\nRunning Vision Critic on retry image...\n")
        try:
            vr2 = vision_review(concept, plan.revised_image_prompt, plan.revised_negative_prompt, str(retry_path))
        except Exception as e:
            raise SystemExit(f"[Vision critic (retry) error] {e}")

        # ── Comparison ───────────────────────────────────────────
        def _delta(a: int, b: int) -> str:
            diff = b - a
            if abs(diff) <= 0.5:
                return "≈ equal"
            return f"↑ +{diff}" if diff > 0 else f"↓ {diff}"

        print("\n" + "=" * 60)
        print("BEFORE / AFTER COMPARISON")
        print("=" * 60)
        rows = [
            ("Overall",               vr.overall_score,               vr2.overall_score),
            ("Composition",           vr.composition_score,           vr2.composition_score),
            ("Print quality",         vr.print_quality_score,         vr2.print_quality_score),
            ("Commercial appeal",     vr.commercial_appeal_score,     vr2.commercial_appeal_score),
            ("Collection consistency",vr.collection_consistency_score,vr2.collection_consistency_score),
        ]
        print(f"  {'Metric':<26} {'Original':>8}  {'Retry':>5}  {'Change'}")
        print(f"  {'-'*26} {'-'*8}  {'-'*5}  {'-'*10}")
        for label, s1, s2 in rows:
            print(f"  {label:<26} {s1.score:>6}/10  {s2.score:>3}/10  {_delta(s1.score, s2.score)}")
        print(f"\n  Original recommendation: {vr.final_recommendation}")
        print(f"  Retry recommendation:    {vr2.final_recommendation}")
        print("=" * 60)
    else:
        print("─" * 60)

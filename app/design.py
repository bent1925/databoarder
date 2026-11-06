"""How did people end up with the change? The one question the data can't answer by itself.

The answer picks the method AND sets how strongly the result may be worded.
"""
from __future__ import annotations

DESIGNS: dict[str, dict[str, str]] = {
    "random": {"emoji": "🎲", "title": "Random split", "sub": "A coin flip decided", "analysis": "ab_test",
               "label": "Random split"},
    "one_date": {"emoji": "📅", "title": "Everyone, from a date", "sub": "Same change for all, one day",
                 "analysis": "before_after", "label": "Everyone, from a date"},
    "one_date_comparison": {"emoji": "⚖️", "title": "From a date, not for all", "sub": "Some never got it",
                            "analysis": "diff_in_diff", "label": "From a date, with a comparison group"},
    "chose": {"emoji": "🙋", "title": "They chose", "sub": "Opted in themselves",
              "analysis": "observational", "label": "People chose"},
    "unsure": {"emoji": "🤷", "title": "Not sure", "sub": "Treated as 'they chose'",
               "analysis": "observational", "label": "Not sure how it was decided"},
}


def design_of(spec) -> str:
    """The design a spec stands for (what the person told us when they picked it)."""
    t = spec.analysis_type
    if t == "ab_test":
        return "random"
    if t == "before_after":
        return "one_date"
    if t == "diff_in_diff":
        return "one_date_comparison"
    return getattr(spec, "assignment", "chose")

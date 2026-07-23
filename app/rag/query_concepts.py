"""Shared bilingual concept expansions for retrieval and answerability."""

from __future__ import annotations

import re
from collections.abc import Iterable

_HAN = re.compile(r"[\u3400-\u9fff]")

CONCEPT_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "affect": ("affects", "affected", "affecting", "influence", "影响", "取决于", "改变"),
    "affected": ("affect", "affects", "affecting", "influence", "影响", "取决于", "改变"),
    "affecting": ("affect", "affects", "affected", "influence", "影响", "取决于", "改变"),
    "affects": ("affect", "affected", "affecting", "influence", "影响", "取决于", "改变"),
    "application": ("applications", "催化", "应用", "用途"),
    "applications": ("application", "催化", "应用", "用途"),
    "catalysis": ("catalyst", "catalytic", "催化"),
    "catalyst": ("catalysis", "catalytic", "催化"),
    "catalytic": ("catalysis", "catalyst", "催化"),
    "determine": ("determines", "determined", "determining", "决定", "取决于", "影响"),
    "determined": ("determine", "determines", "determining", "决定", "取决于", "影响"),
    "determines": ("determine", "determined", "determining", "决定", "取决于", "影响"),
    "determining": ("determine", "determines", "determined", "决定", "取决于", "影响"),
    "density": ("密度", "数密度"),
    "diameter": ("直径", "粒径"),
    "exsolution": ("析出",),
    "evidence": ("proof", "证据", "支持"),
    "factor": ("factors", "因素", "原因", "机制"),
    "factors": ("factor", "因素", "原因", "机制"),
    "influence": ("influences", "influenced", "influencing", "affect", "影响", "取决于"),
    "influenced": ("influence", "influences", "influencing", "affect", "影响", "取决于"),
    "influences": ("influence", "influenced", "influencing", "affect", "影响", "取决于"),
    "influencing": ("influence", "influences", "influenced", "affect", "影响", "取决于"),
    "morphology": ("形貌",),
    "particle": ("颗粒",),
    "particles": ("颗粒",),
    "proof": ("evidence", "证据", "支持"),
    "properties": ("property", "性质", "特性", "性能"),
    "property": ("properties", "性质", "特性", "性能"),
    "stability": ("稳定", "稳定性"),
    "use": ("used", "uses", "应用", "用途"),
    "used": ("use", "uses", "应用", "用途"),
    "uses": ("use", "used", "应用", "用途"),
    "应用": ("用途",),
    "性能": ("性质", "特性"),
    "特性": ("性质", "性能"),
    "性质": ("特性", "性能"),
    "用途": ("应用",),
    "价值": ("催化", "应用", "用途", "性能", "意义"),
    "原因": ("因素", "机制", "影响", "导致"),
    "证明": ("不能", "判断", "支持", "确认", "证据"),
    "直接": ("不能", "仅凭"),
    "给出": ("回答", "判断", "提出", "确认"),
    "结论": ("判断", "确认", "证明"),
    "越多": ("数量", "增加", "减少"),
    "越高": ("性能", "提高", "降低"),
    "有关": ("取决于", "影响", "改变", "因素"),
}


def cjk_expansions(terms: Iterable[str]) -> tuple[str, ...]:
    """Return unique Han-script alternatives for normalized ASCII terms."""

    return tuple(
        dict.fromkeys(
            expansion
            for term in terms
            for expansion in CONCEPT_EXPANSIONS.get(term, ())
            if _HAN.search(expansion)
        )
    )

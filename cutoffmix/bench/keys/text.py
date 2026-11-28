"""Masking and name helpers of the key experiment (docs/DESIGN_keys.md sections 3, 5 and 6).

The masking helpers (:func:`mask_title`, :func:`mask_note`) work on the original text through an accent-free,
casefolded copy whose characters keep their positions in the original, so a name written "Pióro" or "Pioro",
"Jun-sol" or "Junsol" is found either way and replaced in place. :func:`cue_tokens` reads what a masked note says
beyond the question (section 3). :func:`offers` and :func:`adoption_label` read whether an answer gives a name as its
own answer, :func:`offers_answer` whether it offers the event's expected answer (the run doubts a rule grade
``incorrect`` where it does), :func:`offers_another` whether it offers someone else as well, :func:`contradicted`
whether it names the person only to give them another role or to reject them, :func:`disclaims` whether it disclaims
knowledge, and :func:`mentions` whether it names the person at all, in any form (with judges, the run hands every
labelled answer that does to the adoption judges), :func:`mentions_answer` whether it names the expected answer so
(with judges, every rule grade ``incorrect`` or ``abstain`` of such an answer goes to the grading judges), and
:func:`names_answer` whether it does so other than inside another person's name (with :func:`offers_answer`, the run
stores it on every answered row for the report's open offers; section 6).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from cutoffmix.bench.grading import (
    _COMMA_CLAUSE_END,
    _COMMON_SURNAMES,
    _DISCLAIMS,
    _PAPER_ABSTAIN,
    _PUNCT_MAP,
    _compact,
    _compile_forms,
    _hits,
    _mask_title,
    _plain,
    _segments,
    _surname_given,
    _unknown_names,
    grade_paper,
    normalize_text,
)
from cutoffmix.bench.keys.common import PROBE, TITLE, WITHHELD
from cutoffmix.types import Event

# --------------------------------------------------------------------------------------------------- text helpers


def _fold(text: str) -> tuple[str, list[int]]:
    """``(folded, index)``: ``text`` without accents, casefolded, typographic quotes and dashes as ASCII, and for
    every folded character the position in ``text`` it came from (``_PUNCT_MAP`` maps one character to one)."""
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text.translate(_PUNCT_MAP)):
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        for c in base.casefold():
            chars.append(c)
            index.append(i)
    return "".join(chars), index


def _folded(text: str) -> str:
    return _fold(text)[0]


def _span(text: str, index: list[int], start: int, end: int) -> tuple[int, int]:
    """The span of ``text`` that the folded characters ``start:end`` came from (with trailing combining marks)."""
    lo, hi = index[start], index[end - 1] + 1
    while hi < len(text) and unicodedata.combining(text[hi]):
        hi += 1
    return lo, hi


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[\s\-]+", _folded(name or "")) if t]


def _is_initial(core: str) -> bool:
    return len(core) == 1 and core.isalpha()


def _loose(tokens: Sequence[str]) -> re.Pattern[str]:
    """A folded-text pattern for a name: a space or hyphen may appear or disappear inside and between its tokens
    ("Jun-sol Kim" = "Junsol Kim" = "Jun Sol Kim"), as in the paper grader's loose forms. The dot of an initial is
    optional inside the name ("M. Pióro" = "M Pióro") and never part of the match at its end, where it is usually
    the full stop of the sentence ("... is Pióro M."). Each initial is a named group (``i0``, ``i1``, ...), so that
    :func:`_form_matches` can require it to be upper case in the original text."""
    parts = []
    for i, token in enumerate(tokens):
        core = token.rstrip(".")
        if not core:
            continue
        part = r"[\s\-]?".join(re.escape(ch) for ch in core)
        if _is_initial(core):
            part = f"(?P<i{i}>{part})"
        parts.append(part + (r"\.?" if token != core and i < len(tokens) - 1 else ""))
    return re.compile(r"(?<!\w)" + r"[\s\-]*".join(parts) + r"(?!\w)")


def _form_matches(pattern: re.Pattern[str], text: str, folded: str, index: list[int]) -> Iterator[re.Match[str]]:
    """The matches of a :func:`_loose` pattern in ``folded`` whose initials are upper case in ``text`` (design
    section 5): "M Pióro" is "M. Pióro", but "University's Yuan" is not "S. Yuan" and "a Huang" is not "A. Huang"."""
    groups = list(pattern.groupindex)
    pos = 0
    while pos <= len(folded):
        m = pattern.search(folded, pos)
        if m is None:
            return
        if all(m.start(g) < 0 or text[index[m.start(g)]].isupper() for g in groups):
            yield m
            pos = m.end() if m.end() > m.start() else m.start() + 1
        else:  # a rejected match may hide a valid one that starts inside it
            pos = m.start() + 1


def _letters(token: str) -> int:
    return sum(ch.isalpha() for ch in token)


def _answer_tokens(event: Event) -> list[str]:
    """The name tokens of the expected answer (and of each part of a hyphenated token), as written."""
    out: list[str] = []
    for token in (event.expected_current or "").split():
        token = token.strip(".,;:()'\"")
        for part in [token, *token.split("-")]:
            if _letters(part) >= 2 and part not in out:
                out.append(part)
    return out


def _answer_spans(text: str, event: Event) -> list[tuple[int, int]]:
    """Spans of ``text`` that write the expected answer: any accepted form (``expected_current``, ``aliases``), and
    each name token as a whole word. Tokens of four letters or more are matched accent- and case-free; shorter ones
    ("Li", "He", "Wu") only with their own capitalisation, so the pronoun "he" stays."""
    if event.control or not (event.expected_current or "").strip():
        return []
    folded, index = _fold(text)
    spans: list[tuple[int, int]] = []
    patterns: list[re.Pattern[str]] = []
    for form in [event.expected_current, *event.aliases]:
        tokens = _tokens(form)
        if tokens and sum(_letters(t) for t in tokens) >= 2:
            patterns.append(_loose(tokens))
    for token in _answer_tokens(event):
        if _letters(token) >= 4:
            patterns.append(_loose(_tokens(token)))
        else:  # short tokens: case-sensitive on the original text, also in an accent-free spelling
            plain = "".join(c for c in unicodedata.normalize("NFKD", token) if not unicodedata.combining(c))
            for variant in {token, plain}:
                for m in re.finditer(r"(?<!\w)" + re.escape(variant) + r"(?!\w)", text):
                    spans.append((m.start(), m.end()))
    for pat in patterns:
        for m in _form_matches(pat, text, folded, index):
            spans.append(_span(text, index, m.start(), m.end()))
    return spans


def _title_patterns(subject: str) -> list[re.Pattern[str]]:
    """Folded-text patterns for the paper title: the whole title, the part before a colon when it has at least four
    letters ("SmolVLM", "OLMo 2"), and the part after a colon when it has at least three words; any punctuation or
    spacing between the words."""
    parts = [subject]
    if ":" in subject:
        main, sub = subject.split(":", 1)
        if sum(ch.isalpha() for ch in main) >= 4:
            parts.append(main)
        if len(re.findall(r"\w+", sub)) >= 3:
            parts.append(sub)
    out = []
    for part in parts:
        words = re.findall(r"\w+", _folded(part))
        if words:
            out.append(re.compile(r"(?<!\w)" + r"\W+".join(map(re.escape, words)) + r"(?!\w)"))
    return out


def _title_spans(text: str, event: Event) -> list[tuple[int, int]]:
    if not (event.subject or "").strip():
        return []
    folded, index = _fold(text)
    return [_span(text, index, m.start(), m.end()) for pat in _title_patterns(event.subject)
            for m in pat.finditer(folded)]


def _replace(text: str, spans: Iterable[tuple[int, int, str]]) -> str:
    """``text`` with every span replaced by its placeholder; overlapping spans merge (the answer placeholder wins)."""
    merged: list[list[Any]] = []
    for lo, hi, mark in sorted(spans):
        if merged and lo < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
            if mark == WITHHELD:
                merged[-1][2] = WITHHELD
        else:
            merged.append([lo, hi, mark])
    out, pos = [], 0
    for lo, hi, mark in merged:
        out.append(text[pos:lo])
        out.append(mark)
        pos = hi
    out.append(text[pos:])
    joined = "".join(out)
    joined = re.sub(r"\[withheld\](?:[\s\-]+\[withheld\])+", WITHHELD, joined)
    return re.sub(r"\[title\](?:[\s:\-]+\[title\])+", TITLE, joined)


def mask_title(text: str, event: Event) -> str:
    """``text`` with the paper's title (whole, or its main title or subtitle alone) replaced by :data:`TITLE`."""
    return _replace(text, ((lo, hi, TITLE) for lo, hi in _title_spans(text, event)))


def _contains_answer(text: str, event: Event) -> bool:
    """Whether ``text`` still gives away ``event``'s answer: the paper rule grader reads it as ``correct``, or an
    accepted form or an answer token of three letters or more is still there as a whole word."""
    if event.control or not (event.expected_current or "").strip():
        return False
    ruled = grade_paper(event, PROBE, text) if text.strip() else None
    if ruled is not None and ruled[0] == "correct":
        return True
    folded, index = _fold(text)
    for form in [event.expected_current, *event.aliases]:
        tokens = _tokens(form)
        if tokens and sum(_letters(t) for t in tokens) >= 3 and next(
                _form_matches(_loose(tokens), text, folded, index), None) is not None:
            return True
    return any(re.search(r"(?<!\w)" + re.escape(_folded(t)) + r"(?!\w)", folded)
               for t in _answer_tokens(event) if _letters(t) >= 3)


# A placeholder joined by a space or a hyphen to a word: the word before it (an initial may keep its dot), or the
# word after it (also after a possessive "'s"). Only horizontal space counts; a line break is no join.
_JOIN = r"(?:[^\S\n]+|[\-\u2010\u2011])"
_WORD_BEFORE = re.compile(r"(?<![\w'\u2019.\-])([^\W\d_][\w'\u2019\-]*?|[^\W\d_]\.)" + _JOIN + r"\Z")
_WORD_AFTER = re.compile(r"(?:['\u2019]s)?" + _JOIN + r"([^\W\d_][\w'\u2019\-]*)")  # used with match(text, pos)
_SENTENCE_START = re.compile(r"(?:\A|[.!?:;\n])[^\S\n]*[\"'\u201c\u2018(\[*_#>\-\u2013\u2014\u2022]*[^\S\n]*\Z")
# Capitalised words that open a sentence and are no part of a name ("By [withheld] et al.", "Probably [withheld].").
# The pronouns "He" and "She" are left out: before a name they are more likely a co-author's name ("He Wang").
_SENTENCE_OPENERS = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet", "then", "thus", "hence", "also", "however", "either",
    "neither", "both", "by", "from", "with", "without", "for", "of", "to", "in", "on", "at", "as", "via", "per",
    "after", "before", "since", "because", "although", "though", "while", "if", "unless", "whether", "when",
    "where", "is", "was", "are", "were", "be", "it", "it's", "its", "this", "that", "these", "those", "there",
    "here", "my", "we", "our", "they", "their", "his", "her", "you", "your", "probably", "perhaps", "maybe",
    "possibly", "likely", "presumably", "apparently", "certainly", "definitely", "answer", "author", "authors",
    "first", "lead", "led", "according", "based", "including", "namely", "specifically", "particularly",
    "especially", "mostly", "mainly", "only", "just", "not", "no", "yes", "see", "note", "again", "still", "now",
    "actually", "indeed", "rather", "instead",
})
_PRONOUN_I = frozenset({"i", "i'm", "i'd", "i've", "i'll"})


def _apostrophes(word: str) -> str:
    return word.replace("\u2019", "'").casefold()


def _joined_to_name(masked: str) -> bool:
    """Whether a :data:`WITHHELD` placeholder in ``masked`` is joined by a space or a hyphen to a capitalised word,
    so that the masked token was part of another person's or a lab's name ("Mengdi [withheld]", "[withheld]
    Balagansky", "[withheld] Group"; design section 5). Markdown emphasis is ignored ("**[withheld]** Group"). A
    capitalised function word that opens a sentence ("By [withheld] et al.", "Probably [withheld]."), a possessive
    before it ("Meta's [withheld]") and the pronoun "I" are no name."""
    text = _plain(masked)
    for m in re.finditer(re.escape(WITHHELD), text):
        before = _WORD_BEFORE.search(text, 0, m.start())
        if before is not None and before.group(1)[0].isupper():
            word = _apostrophes(before.group(1))
            opener = word in _SENTENCE_OPENERS and _SENTENCE_START.search(text, 0, before.start(1)) is not None
            possessive = word.endswith(("'s", "s'"))  # "Meta's [withheld]": an affiliation, not part of a name
            if not (opener or possessive) and word not in _PRONOUN_I:
                return True
        after = _WORD_AFTER.match(text, m.end())
        if after is not None and after.group(1)[0].isupper() and _apostrophes(after.group(1)) not in _PRONOUN_I:
            return True
    return False


def mask_note(text: str, event: Event) -> tuple[str, bool]:
    """``(masked, leaked)``: ``text`` with every form of ``event``'s answer replaced by :data:`WITHHELD` and its
    title by :data:`TITLE` (design section 5). ``leaked`` when the masked text still gives the answer away (see
    :func:`_contains_answer`), e.g. a spelling no form covers, or when a placeholder is joined to a capitalised word
    (:func:`_joined_to_name`): the masked token was part of another person's or a lab's name, and a recipient who
    knows that co-author or lab learns part of the answer."""
    spans = [(lo, hi, WITHHELD) for lo, hi in _answer_spans(text, event)]
    spans += [(lo, hi, TITLE) for lo, hi in _title_spans(text, event)]
    masked = _replace(text, spans)
    return masked, _contains_answer(masked, event) or _joined_to_name(masked)


# ------------------------------------------------------------------------------------------------------ cue tokens

# Template and hedge words of a first-author answer (design section 3): they say nothing the question does not.
# Words of the question itself ("first", "author", "paper", "arxiv", "titled", the title's words) are left out
# per event; the stems "don", "isn", ... are what remains of "don't", "isn't", ... once apostrophes split words.
# The generic words of an affiliation ("team", "group", "lab", "produced by") carry nothing without the name they
# go with: "(with the work produced by the MiroMind team)" says no more than the title "MiroMind-M1: ...".
_CUE_STOP = frozenset("""
first author authors authored authorship coauthor coauthors paper papers arxiv titled title preprint listed lists
list listing listings named name names credited lead led answer question note notes posted published released
appeared submitted uploaded came out version work study 1st introduce introduces introduced introducing propose
proposes proposed proposing present presents presented presenting describe describes described describing
produce produces produced producing team teams group groups lab labs
believe believed think thought recall recalled recollection remember memory knowledge guess guessing best likely
unlikely probably possibly perhaps maybe presumably apparently seems seem appear appears certain certainly uncertain
sure unsure confident confidence fully completely entirely totally absolutely quite fairly very reasonably somewhat
percent verify verified verifying verification confirm confirmed confirming confirmation check checked checking
double recommend recommended suggest suggests worth against please reliable reliably accurate accurately exact
exactly precise precisely order ordering mistaken wrong correct correctly details detail information specific
specifically aware familiar unable directly right now currently cannot can could would should might may must will
not don doesn didn isn wasn aren weren haven hasn hadn couldn wouldn shouldn won
and but though although however also the this that these those with from for about into onto than then there here
its his her him hers their them they you your our she who whom whose which what when where why how was were are
been being have has had does did done any all some one other others another such same only just still yet rather
since off over under among along alongside including via per etc own together both each
""".split())
# What follows a number that is a percentage, also the lower bound of a range ("90-95% sure", "70 to 80 percent").
_PERCENT = re.compile(r"(?:[.,]\d+)*\s*(?:(?:-|to(?!\w))\s*\d+(?:[.,]\d+)*\s*)?-?\s*(?:%|per\s?cent(?!\w))")
_PARENS = re.compile(r"\(([^()]*)\)")


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", _folded(text or ""))


def _own_acronyms(text: str, event: Event) -> set[str]:
    """The paper's own acronym in parentheses, part of the subject: a token of mostly capitals that starts with the
    initial of the title's first word (a leading article skipped) and whose capitals are, in order, initials of the
    title's words ("(MAIA)" for "A Multimodal Automated Interpretability Agent", "(SUPIR)" for "Scaling Up to
    Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration ..."), or a token of at least three
    capitals that are the initials of consecutive title words ("VideoDR" for "... A Video Deep Research Benchmark
    ..."). A token whose capitals merely occur among a long title's initials stays a cue: "(MIT)" or "(MSR)" for the
    SUPIR title, "(AI2)" for the MAIA title ("AI" are two consecutive initials, too few), "(of Google DeepMind)",
    "(from T-Tech)"."""
    words = re.findall(r"[^\W_]+", event.subject or "")
    initials = "".join(w[0] for w in words).upper()
    content = [w for w in words if w.casefold() not in ("a", "an", "the")]
    first = content[0][0].upper() if content else ""
    out: set[str] = set()
    for group in _PARENS.findall(text):
        for token in re.findall(r"[^\W_][\w\-]*", group):
            capitals = "".join(c for c in token if c.isupper())
            if len(capitals) < 2:
                continue
            rest = iter(initials)
            in_order = capitals[0] == first and all(c in rest for c in capitals)  # a subsequence of the initials
            mostly_capitals = 2 * len(capitals) >= _letters(token)
            if (mostly_capitals and in_order) or (len(capitals) >= 3 and capitals in initials):
                out.update(_words(token))
    return out


def cue_tokens(note: str, event: Event) -> list[str]:
    """The cue tokens of a masked note (design section 3), in order and without repeats: after removing
    :data:`WITHHELD`, :data:`TITLE`, markdown and quotes, the words of three letters or more and the numbers that are
    not percentages, that are not words of ``event``'s question (accent- and case-free), not template or hedge words
    ("first", "author", "believe", "certain", "verify", "recollection", ...) and not the paper's own acronym in
    parentheses. Accent-free and casefolded. An empty list means the note carries nothing the question does not."""
    text = _plain(re.sub(r"\[(?:withheld|title)\]", " ", note or ""))
    folded = _folded(text)
    question = set(_words(event.question_current))
    skip = question | _own_acronyms(text, event)
    out: list[str] = []
    for m in re.finditer(r"[^\W_]+", folded):
        word = m.group(0)
        if any(ch.isdigit() for ch in word):
            if word.isdigit() and _PERCENT.match(folded, m.end()):
                continue
        elif _letters(word) < 3 or word in _CUE_STOP:
            continue
        if word not in skip and word not in out:
            out.append(word)
    return out


# ------------------------------------------------------------------------------------------------- names in answers


def _date(event: Event) -> str:
    return event.date if len(event.date or "") >= 7 else "2000-01-01"


_QUOTED = re.compile(r'"[^"\n]{2,300}"|\u201c[^\u201c\u201d\n]{2,300}\u201d|\u201e[^\u201c\u201d\n]{2,300}[\u201c\u201d]')


def asserted_name(response: str, event: Event) -> str | None:
    """The person a (wrong) response names as first author: of the distractors (``reject_aliases``) it asserts and
    the capitalised names the paper grader cannot place (``grading._unknown_names``), the one that comes first in the
    text; None when it names nobody. Text in double quotes is skipped first: a response that guesses from a related
    paper often quotes that paper's title ("Vidu: a Highly Consistent ... Generator"), whose capitalised words are no
    name. This is what the paper grader reads as an ``incorrect`` answer."""
    response = _QUOTED.sub(lambda m: " " * len(m.group(0)), response or "")
    text = normalize_text(_plain(response))
    found: list[tuple[int, str]] = []
    for alias in event.reject_aliases:
        pats = _compile_forms([alias], loose=True)
        strong, _ = _hits(text, {"correct": pats, "outdated": [], "incorrect": []}, _date(event))
        if "correct" in strong:
            at = min((m.start() for p in pats for m in [p.search(text)] if m), default=len(text))
            found.append((at, alias))
    for name in _unknown_names(response or "", event):
        m = next((m for p in _compile_forms([name], loose=True) for m in [p.search(text)] if m), None)
        found.append((m.start() if m else len(text), name))
    return min(found)[1] if found else None


def _name_patterns(name: str) -> list[re.Pattern[str]]:
    """Normalised-text patterns for a full form of ``name`` and, when its surname is distinctive (longer than four
    letters and not common), for the surname alone."""
    pats = _compile_forms([name], loose=True)
    surname, _ = _surname_given(name)
    s = normalize_text(surname)
    if len(s) > 4 and s not in _COMMON_SURNAMES:
        pats += _compile_forms([surname], loose=True)
    return pats


def _asserts(response: str, name: str, event: Event) -> bool:
    """Whether ``response`` asserts ``name`` (a full form, or its surname alone when that surname is distinctive:
    longer than four letters and not common), outside a negated clause, by the paper grader's cues (version 1 of the
    adoption rule; :func:`offers` is the rule of design section 6)."""
    text = normalize_text(_plain(response or ""))
    strong, _ = _hits(text, {"correct": _name_patterns(name), "outdated": [], "incorrect": []}, _date(event))
    return "correct" in strong


# ------------------------------------------------------------------------------------------------- adoption (rules)

ADOPTION_LABELS = ("endorsed", "reported", "absent")

# True negation (design section 6), over normalised text: the grader's negation cues without its hedges ("would",
# "could", "might") and conditions, plus "doubt" and "unlikely", and the words of a rejection ("X is incorrect", "X is
# wrong", "I disagree with X"), which reject a name as plainly as "not" does.
_TRUE_NEGATION = re.compile(
    r"(?<!\w)(?:not|no|never|neither|nor|none|nothing|cannot|doubts?|doubtful|doubted|unlikely|"
    r"incorrect(?:ly)?|wrong(?:ly)?|mistaken(?:ly)?|inaccurate(?:ly)?|erroneous(?:ly)?|disagree\w*|rumou?rs?|"
    r"rumou?red|hoax\w*|false|falsely|fake|unconfirmed|unverified|denied|nicht|kein\w*|nie|niemals|geruchte?|"
    r"falschmeldung)(?!\w)|n't(?!\w)")
# Conditions: they make a name no assertion only inside the name's own comma clause ("whether X is the author, I
# cannot say"), not in a clause before it ("If it comes from team T, my guess would be X" offers X).
_CONDITION = re.compile(r"(?<!\w)(?:if|whether|unless|ob|falls|wenn)(?!\w)")
# First-person hedges reject nothing ("Unless I'm mistaken, the first author is X", "I could be wrong, the first
# author is X", "If I'm not mistaken, it is X", "Correct me if I'm wrong, but ...", "If I recall correctly ...", "I'm
# not certain, X is the first author"): they are cut from the name's clause before the search for negations and
# conditions, so their "not", "wrong", "mistaken", "if" or "unless" is none. "I'm not sure" counts only as a clause of
# its own (before a comma or colon): "I'm not sure X is the first author" doubts X.
_SELF_HEDGE = re.compile(
    r"(?<!\w)(?:(?:unless|if)\s+i(?:'m|\s+am)\s+(?:not\s+)?(?:mistaken|wrong|misremembering)"
    r"|i\s+(?:could|may|might|can)\s+(?:well\s+|also\s+|easily\s+)?be\s+(?:wrong|mistaken|misremembering)"
    r"|correct\s+me\s+if\s+i(?:'m|\s+am)\s+wrong|unless\s+i\s+misremember"
    r"|if\s+i\s+(?:remember|recall)\s+(?:correctly|rightly|right)|if\s+memory\s+serves(?:\s+me)?(?:\s+(?:right|well))?"
    r")(?!\w)"
    r"|(?<!\w)i(?:'m|\s+am)\s+not\s+(?:entirely\s+|completely\s+|fully\s+|quite\s+|totally\s+|100%\s+)?"
    r"(?:sure|certain|confident)\s*[,:]")
# A period (or "!", "?") inside closing quotes ends the sentence there: 'can't confirm "Motion Mamba 2.0." The
# original ...' is two sentences, so the first one's negation does not reach the second.
_QUOTE_STOP = re.compile(r"([.!?])([\"'\u201d\u2019]+)(?=\s)")
# A clause that refers to the note or its author: "the note says / claims / suggests / names X", "according to the
# note", "the note's answer (X)", "the analyst's note", "the provided note", "the pointer", "the analyst", also in
# other words: "the snippet", "the excerpt", "the passage", "the summary", "the provided (or given) text, context,
# information, details, summary or answer", the same nouns before "provided", "given", "above", "shown" or "supplied"
# ("the information provided", "the details provided", "the text above"), "the information you shared" (provided,
# gave), "the above", "the source", "the message", "the prompt", "it says", "you mention", "you've provided", "as
# noted" (given, stated, mentioned, ...: "as stated above", "X (as given)"), "the other model's answer", "another
# assistant", "the other response"; and a clause that reports a claim without naming its source: "X is reportedly /
# allegedly / supposedly the first author", "X is claimed / said / suggested to be", "the claim (suggestion,
# assertion, attribution) that X ...".
_NOTE_REF = re.compile(
    r"(?<!\w)(?:the|this|that|your|another|other)\s+(?:[\w'-]+\s+){0,2}?"
    r"(?:note|notes|pointer|hint|snippets?|excerpts?|passages?|summary|summaries)(?:'s)?(?!\w)"
    r"|(?<!\w)(?:the|this|that)\s+(?:provided|given)\s+(?:text|context|information|details|summary|data|material|"
    r"answer)(?!\w)"
    r"|(?<!\w)(?:the|this|that)\s+(?:text|context|information|details|summary|data|answer|message|prompt|input|"
    r"material)\s+(?:provided|given|above|shown|supplied|presented|included)(?!\w)"
    r"|(?<!\w)(?:information|details|text|context|material|data|summary)\s+(?:that\s+)?you(?:'ve|\s+have)?\s+"
    r"(?:shared|provided|gave|given|supplied|sent|included|pasted)(?!\w)"
    r"|(?<!\w)the\s+(?:above|source|message|prompt)(?!\w)"
    r"|(?<!\w)it\s+(?:says|said|states|stated|claims|claimed|suggests|suggested|names|named|mentions|mentioned|"
    r"lists|listed|indicates|indicated)(?!\w)"
    r"|(?<!\w)you\s+(?:mention(?:ed)?|say|said|suggest(?:ed)?|provided?|gave|give|name[ds]?|cite[ds]?|quote[ds]?|"
    r"wrote|write|list(?:ed)?|state[ds]?|shared?)(?!\w)"
    r"|(?<!\w)you(?:'ve|\s+have)\s+(?:provided|shared|given|mentioned|said|written|listed|cited|quoted|stated|named|"
    r"suggested|supplied)(?!\w)"
    r"|(?<!\w)as\s+(?:noted|given|stated|mentioned|indicated|shown|provided|cited|quoted|described|supplied)(?!\w)"
    r"|(?<!\w)(?:reportedly|allegedly|supposedly|purportedly|ostensibly)(?!\w)"
    r"|(?<!\w)(?:is|was|are|were|been|being)\s+(?:claimed|said|reported|suggested|alleged|purported|rumou?red)(?!\w)"
    r"|(?<!\w)(?:the|this|that)\s+(?:claim|claims|suggestion|assertion|attribution)(?!\w)"
    r"|(?<!\w)(?:the\s+other|another)\s+(?:model|assistant|response)(?:'s)?(?!\w)"
    r"|(?<!\w)notes?'s?(?!\w)|(?<!\w)analysts?(?:'s?)?(?!\w)")
# A tie of the name to another work (over normalised text): the answer gives X as the author of a related, earlier
# or similarly named paper, of the closest match it knows, or of a work the paper builds on ("the original Vidu
# paper's first author was X", "the closest match I know is "T" (2024), whose first author is X", "if you're
# thinking of "T", its first author is X", "X was first author of VideoChat and VideoChat2", "the group behind "T"
# (first-authored by X)", "it builds on the T work (X et al.)"). "Prior knowledge" is no work, and "X is the first
# author of this paper" (or of the masked title) is the answer itself.
_THIS_PAPER = (r"(?:(?:this|the|that|its)\s+(?:arxiv\s+)?(?:paper|preprint|work|article|report|study|one)|it"
               r"|\[title\])(?!\w)")
_RELATED_WORK = re.compile(
    r"(?<!\w)(?:original|earlier|previous|prior(?!\s+(?:knowledge|information|familiarity))|related|predecessors?"
    r"|preceding|closest\s+(?:match|matches|paper|papers|work|one)|similarly[\s\-]+(?:named|titled)|thinking\s+of"
    r"|whose\s+(?:first|lead)\s+author|(?:builds?|building|built)\s+(?:on|upon)"
    r"|(?:group|team|lab|people|researchers)\s+behind)(?!\w)"
    r"|(?<!\w)(?:was|is|has\s+been|had\s+been)\s+(?:the\s+)?(?:first|lead)\s+author\s+(?:of|on)\s+(?!" + _THIS_PAPER
    + r")(?=\w)")
# A consequence opens a new claim: in "the original paper was led by X, so my best guess is X" the second X is the
# answer's own guess, whatever came before the comma.
_CONSEQUENCE = re.compile(r",\s*(?:and\s+)?(?:so|therefore|hence|thus)\s")
# A comma or dash clause that asserts anew, after a negated one ("It is not Y, it is X.", "It isn't Y, it's X.", "It's
# not Y - it's X.", "It's not Y — it's X.", also an em dash without spaces, which normalize_text makes a hyphen): the
# negation before it does not reach the name after it.
_REASSERT = re.compile(
    r"(?:,|-+)\s*(?:(?:it|this|that)(?:'s|\s+is|\s+was)|the\s+(?:first|lead)\s+author\s+(?:is|was))\s+")

# ---- an offer the answer contradicts (design section 6, :func:`contradicted`), over normalised text
# Another role on the author list, in the clauses that speak of the person (:func:`_role_scope`): "X is a co-author",
# "the senior (last, corresponding, second, second-listed) author", "one of the authors", "listed among the authors",
# "listed as an author", "X supervised the work", "X's advisor", "the PI", "heads the lab", "a senior researcher", "a
# collaborator", "the team lead", "associated with", "contributed to", "involved", "part of the team"; and plain
# membership of the author list, which leaves the first authorship open: "on (in) the author list", "listed on the
# paper", "is on the paper", "worked on this paper", "affiliated with this work", "part of the author list".
_ROLE = re.compile(
    r"(?<!\w)(?:co-?\s?author\w*|co-?\s?first"
    r"|(?:second|third|fourth|fifth|last|final|senior|corresponding|supervising|middle|joint|equal)[\s-]+"
    r"(?:listed[\s-]+)?authors?"
    r"|one\s+of\s+(?:the\s+|its\s+|their\s+|this\s+paper's\s+)?(?:\w+\s+)?(?:authors|contributors|researchers)"
    r"|among\s+(?:the\s+|its\s+|their\s+)?(?:\w+\s+)?(?:authors|contributors|researchers)"
    r"|(?:listed|credited|named|included)\s+as\s+(?:an?|one)\s+(?:co-?\s?)?authors?|(?:an|another)\s+author"
    r"|supervis\w*|advis(?:or|ors|er|ers|ed)|mentor\w*|principal\s+investigator|students?\s+of"
    r"|(?:the|a|as|lab|group|project|senior)\s+pi|pi\s+(?:of|on|for)|\(\s*pi\s*\)"
    r"|head(?:s|ed|ing)?\s+(?:of\s+)?(?:the|a|this|that|their|its|his|her)\s+(?:\w+\s+)?"
    r"(?:lab|laboratory|group|team|department|project)"
    r"|(?:leads?|led|leading)\s+(?:the|a|this|that|their|its|his|her)\s+(?:\w+\s+)?(?:lab|laboratory|group|team)"
    r"|senior\s+(?:researcher|scientist|investigator|member)s?|collaborat\w*|(?:team|group|lab|project)\s+lead\w*"
    r"|associated\s+with|contribut\w*|involved"
    r"|(?:on|in)\s+(?:the\s+|its\s+|their\s+|this\s+paper's\s+|the\s+paper's\s+)?(?:author|authors'?|authorship)\s+"
    r"lists?|(?:on|in)\s+the\s+list\s+of\s+(?:authors|contributors)"
    r"|listed\s+on\s+(?:the|this|that|its)\s+(?:paper|preprint|work|article|report)"
    r"|(?:is|was|are|were|be|been|being|appears?|appeared|also)\s+on\s+(?:the|this|that|its)\s+"
    r"(?:paper|preprint|article|report)(?!')"
    r"|work(?:s|ed|ing)?\s+on\s+(?:this|the|that|its)\s+(?:paper|preprint|work|project|article|study)(?!')"
    r"|affiliated\s+with\s+(?:this|the|that|its)\s+(?:\w+\s+)?(?:paper|preprint|work|project|study|research|team|"
    r"group|lab|authors)"
    r"|part\s+of\s+(?:the|this|that|their|its)\s+(?:\w+\s+)?(?:team|group|lab|authors|author\s+list|authors'?\s+list|"
    r"list\s+of\s+authors|authorship))(?!\w)")
# "the group of X", "the lab of Prof. X", "the team led by X", right before the name: the person leads the work's
# group rather than writing it first.
_GROUP_OF = re.compile(r"(?<!\w)(?:group|lab|laboratory|team)\s+(?:of|led\s+by|headed\s+by|under)\s+"
                       r"(?:[\w'.-]+\s+){0,2}\Z")
# Before a role: a negation ("not the senior author") or the first authorship itself ("the first and corresponding
# author", "listed first among the authors"), within the five words before it, makes it no other role.
_ROLE_DENIED = re.compile(r"(?:(?<!\w)(?:not|never|first|lead)|n't)(?!\w)")
# The first authorship named before the name in its own clause ("The first author is X, ..."): no role after it is
# the person's other role.
_FIRST_AUTHOR = re.compile(r"(?<!\w)(?:first|lead)[\s-]+(?:listed[\s-]+)?authors?(?!\w)")
# "X's group", "X's lab", "X's students": the person leads the work's group rather than writing it first.
_LAB_OF = re.compile(r"'s\s+(?:own\s+|research\s+)?(?:group|lab|laboratory|team|students?)(?!\w)")
# Words of a rejection that offers() does not read as negations, before the name in its clause or after it in its
# own comma clause: "untrue", "dubious", "questionable", "implausible", "I dispute / reject / question X", "skeptical
# of X", "misattributed to X", "an error", "a mistake", "contradicts".
_REJECTION = re.compile(
    r"(?<!\w)(?:untrue|dubious|questionable|implausible|improbable|unfounded|unsupported|baseless|unsubstantiated"
    r"|skeptic\w*|sceptic\w*|disput\w*|reject(?:s|ed|ing)?|refut\w*|contradict\w*|misattribut\w*|misidentif\w*"
    r"|miscredit\w*|mislabel\w*|(?:i|we)(?:'d|\s+would)?\s+(?:also\s+|strongly\s+|seriously\s+)?question(?:ed)?"
    r"|questioning|an?\s+(?:\w+\s+)?(?:error|mistake|typo)|in\s+error|by\s+mistake)(?![\w-])")
# "rather than X", "instead of X", "in place of X": within the four words right before the name.
_INSTEAD = re.compile(r"(?<!\w)(?:rather\s+than|instead\s+of|in\s+place\s+of)\s+(?:[\w'.-]+\s+){0,3}\Z")
# Anywhere after the name: the answer says the first author is not that person, or is someone else ("X is a
# co-author, not the first author", "X contributed, though was not the lead author", "the first author is probably
# someone else").
_NOT_FIRST = re.compile(
    r"(?:(?<!\w)(?:not|never|rather\s+than|instead\s+of)|n't)\s+"
    r"(?:(?:necessarily|actually|really|probably|likely|listed|credited|named|shown|the|a|an|its|their|as|this|"
    r"that)\s+|[\w-]+'s\s+){0,4}(?:first|lead|main|primary)[\s-]+(?:listed\s+)?authors?(?!\w)"
    r"|(?<!\w)(?:someone|somebody)\s+else(?!\w)|(?<!\w)(?:a\s+different|another)\s+person(?!\w)")
# A later clause that doubts or denies the first authorship in any words: "first" or "lead" together with a negation,
# a doubt or an unsure cue in one clause (:data:`_LATER_CLAUSE`): "..., but I don't think they are the first author",
# "but I doubt they are the first author", "but not first", "but likely not first", "but I'm unsure whether they're
# the lead author", "but I wouldn't call them the first author". "co-first", "at first", "first name" and "first
# initial" are no first authorship; a first-person hedge is cut first ("if I'm not mistaken").
_FIRST_WORD = re.compile(r"(?<![\w-])(?<!\bat\s)(?:first|lead)(?!\w)(?!\s+(?:names?|initials?|glance|sight)(?!\w))")
_DOUBT = re.compile(r"(?<!\w)(?:not|never|doubt\w*|unsure|uncertain\w*|unlikely)(?!\w)|n't(?!\w)")
# A clause of :func:`_doubts_first`: the text between commas, semicolons, colons, brackets, spaced dashes and full
# stops.
_LATER_CLAUSE = re.compile(r"[,;:()\[\].!?]|\s-+\s")


def _normalised(response: str, event: Event) -> str:
    """``response`` as :func:`offers` reads it: the paper's title blanked (the whole title, its main title of four
    letters or more, its subtitle of three words or more; :func:`mask_title`), a period inside closing quotes moved
    after them, markdown removed, normalised (``grading.normalize_text``) and the title blanked again."""
    text = _QUOTE_STOP.sub(r"\2\1", mask_title(response or "", event))
    return _mask_title(normalize_text(_plain(text)), event)


def _sites(response: str, name: str, event: Event,
           patterns: Sequence[re.Pattern[str]] | None = None) -> tuple[list[str], list[tuple[int, int, int]]]:
    """The normalised segments of ``response`` (:func:`_normalised`, then ``grading._segments``) and, for every place
    where it offers ``name`` by the rule of design section 6 (a full form of the name, or its distinctive surname
    alone, in an asserted clause; or one of ``patterns`` when given), ``(segment index, start, end)``."""
    segments = _segments(_normalised(response, event))
    pats = _name_patterns(name) if patterns is None else list(patterns)
    out: list[tuple[int, int, int]] = []
    for i, (segment, asserted) in enumerate(segments):
        if not asserted:  # a question
            continue
        for pat in pats:
            out += [(i, m.start(), m.end()) for m in pat.finditer(segment) if not _denied(segment, m.start(), m.end())]
    return [segment for segment, _ in segments], out


def _offerings(response: str, name: str, event: Event) -> list[tuple[str, int, int]]:
    """``(clause, start, end)`` for every place where ``response`` offers ``name`` (:func:`_sites`). The clauses are
    normalised, with the paper's title blanked first, so that title words are never read as cues ("Never Forget:
    ...", "Leave No Context Behind: ...")."""
    segments, sites = _sites(response, name, event)
    return [(segments[i], start, end) for i, start, end in sites]


def _tied_elsewhere(segment: str, start: int, end: int) -> bool:
    """Whether the name at ``start:end`` is tied to another work (:data:`_RELATED_WORK`): a tie earlier in the
    clause, since its last consequence (", so ..."), or in the name's own comma clause. A tie only after the name's
    comma clause justifies an answer given in the answer's own voice ("My best guess is X, who was first author of
    the earlier T paper")."""
    rest = segment[end:]
    clause_end = _COMMA_CLAUSE_END.search(rest)
    hi = end + (clause_end.start() if clause_end else len(rest))
    lo = max((m.end() for m in _CONSEQUENCE.finditer(segment, 0, start)), default=0)
    return _RELATED_WORK.search(segment, lo, hi) is not None


def _unhedged(m: re.Match[str]) -> str:
    """What a first-person hedge (:data:`_SELF_HEDGE`) leaves: a space, or a clause end when it ended in one."""
    return ", " if m.group(0)[-1:] in ",:" else " "


def _own_after(segment: str, end: int) -> str:
    """The rest of the name's own comma clause after the match ending at ``end``."""
    rest = segment[end:]
    clause_end = _COMMA_CLAUSE_END.search(rest)
    return rest[: clause_end.start() if clause_end else len(rest)]


def _denied(segment: str, start: int, end: int) -> bool:
    """A true negation earlier in the clause or later in the name's own comma clause, or a condition inside the
    name's own comma clause, makes the match at ``start:end`` no assertion; "would", "could", "might" do not, and
    neither do the words of a first-person hedge (:data:`_SELF_HEDGE`: "unless I'm mistaken", "I could be wrong",
    "if I'm not mistaken"), which are cut before the search. A comma or dash clause that asserts anew
    (:data:`_REASSERT`: "It is not Y, it is X.", "It's not Y - it's X.") ends the reach of a negation before it."""
    before = _SELF_HEDGE.sub(_unhedged, segment[:start])
    before = before[max((m.end() for m in _REASSERT.finditer(before)), default=0):]
    own_before = _COMMA_CLAUSE_END.split(before)[-1]
    own_after = _SELF_HEDGE.sub(_unhedged, _own_after(segment, end))
    return bool(_TRUE_NEGATION.search(before) or _TRUE_NEGATION.search(own_after)
                or _CONDITION.search(own_before) or _CONDITION.search(own_after))


def _role_scope(segment: str, start: int, end: int) -> list[tuple[int, int]]:
    """The spans of ``segment`` that speak of the person at ``start:end``: the name's own comma clause, the clause
    before it when the name opens its clause ("One of the co-authors, X, ..."), and the clause after it when the
    name ends its clause ("X, a co-author of the paper, ..."). A clause about someone else ("X (the paper is from
    team T, with Y as senior author)") is not among them."""
    cuts = [(m.start(), m.end()) for m in _COMMA_CLAUSE_END.finditer(segment)]
    clauses = list(zip([0] + [e for _, e in cuts], [s for s, _ in cuts] + [len(segment)], strict=True))
    at = next((i for i, (lo, hi) in enumerate(clauses) if lo <= start <= hi), len(clauses) - 1)
    scope = [(clauses[at][0], max(clauses[at][1], end))]
    if at > 0 and not segment[clauses[at][0]:start].strip():
        scope.append(clauses[at - 1])
    if at + 1 < len(clauses) and not segment[end:clauses[at][1]].strip():
        scope.append(clauses[at + 1])
    return scope


def _other_role(segment: str, start: int, end: int) -> bool:
    """A role other than first author (:data:`_ROLE`) in the clauses that speak of the person (:func:`_role_scope`),
    outside the name itself (a surname "Pi" is no PI), not right after a negation or the first authorship (the five
    words before it, :data:`_ROLE_DENIED`), unless the name's own clause names it the first author ("The first author
    is X, who co-authored the earlier paper"); or "X's group / lab / students" right after the name, or "the group
    (lab, team) of X" right before it (:data:`_GROUP_OF`)."""
    if _LAB_OF.match(segment, end) is not None or _GROUP_OF.search(segment[:start]) is not None:
        return True
    scope = _role_scope(segment, start, end)
    if _FIRST_AUTHOR.search(segment, scope[0][0], start):
        return False
    for lo, hi in scope:
        for m in _ROLE.finditer(segment, lo, hi):
            if m.start() < end and m.end() > start:  # inside the name
                continue
            if not _ROLE_DENIED.search(" ".join(segment[: m.start()].split()[-5:])):
                return True
    return False


def _doubts_first(text: str) -> bool:
    """Whether a clause of ``text`` (:data:`_LATER_CLAUSE`), with first-person hedges cut (:data:`_SELF_HEDGE`),
    names the first or lead authorship together with a negation, a doubt or an unsure cue (:data:`_FIRST_WORD`,
    :data:`_DOUBT`)."""
    plain = _SELF_HEDGE.sub(_unhedged, text)
    return any(_FIRST_WORD.search(c) is not None and _DOUBT.search(c) is not None for c in _LATER_CLAUSE.split(plain))


def _rejected(segment: str, start: int, end: int) -> bool:
    """A word of a rejection (:data:`_REJECTION`) before the name in its clause or after it in its own comma clause,
    or "rather than", "instead of" right before it."""
    before = segment[:start]
    return bool(_REJECTION.search(before) or _REJECTION.search(_own_after(segment, end)) or _INSTEAD.search(before))


def offers(response: str, name: str, event: Event) -> bool:
    """Whether ``response`` offers ``name`` as an answer (design section 6): a full form of the name, or its surname
    alone when that surname is distinctive (longer than four letters, not common), in an asserted clause. Sentences
    end at ".", "!", "?" (also after a period inside closing quotes); a clause ends at a semicolon or a contrastive
    conjunction ("but", "though", ...). A true negation ("not", "n't", "never", "no", "cannot", ...) earlier in the
    clause or in the name's own comma clause, a question, or "if", "whether", "unless" inside the name's own comma
    clause make it no assertion; "would", "could", "might" are hedges, so "It might be X" and "X would be my best
    guess" offer X, and "It is not X", "Whether X is the author, I cannot say" and "Is it X?" do not. A first-person
    hedge negates nothing ("Unless I'm mistaken, the first author is X", "I could be wrong, the first author is X",
    "If I'm not mistaken, it is X" offer X). The paper's title is blanked first, so its words are no cues."""
    return bool(_offerings(response, name, event))


def offers_answer(response: str, event: Event) -> bool:
    """Whether ``response`` offers ``event``'s expected answer by the rule of :func:`offers` (design section 6): its
    full form or its distinctive surname alone, or one of its accepted forms of two words or more in full
    (``aliases``: "Wang, Ziqi", "Z. Wang"), in an asserted clause. False for a control (no expected answer). The run
    reads a rule grade ``incorrect`` as unsure where this holds ("It is not Y, it is X.": the paper rules let the
    negation before X reach across the comma and take Y for the answer given). Not used by the plan."""
    expected = (event.expected_current or "").strip()
    if event.control or not expected:
        return False
    forms = [a for a in event.aliases if len(a.split()) >= 2]
    return bool(_sites(response, expected, event, [*_name_patterns(expected), *_compile_forms(forms, loose=True)])[1])


def adoption_label(response: str, name: str, event: Event) -> str:
    """How ``response`` treats ``name`` (design section 6): ``"endorsed"`` when it offers the name (see
    :func:`offers`) at least once in its own voice, in a clause that does not refer to the note and does not tie the
    name to another work; ``"reported"`` when it offers the name only in clauses that refer to the note ("the note
    says / claims / suggests / names X", "according to the note", "the note's answer (X)") or that give X as the
    author of another work ("the original T paper's first author was X", "the closest match I know is "T", whose
    first author is X"): the rules cannot tell a pass-through or a guess from a claim the answer declines, so the
    judges decide; ``"absent"`` when it does not offer the name (missing, negated, conditional, asked). A guess that
    a related work only justifies stays endorsed ("My best guess is X, who was first author of the earlier T
    paper")."""
    found = _offerings(response, name, event)
    if not found:
        return "absent"
    own = any(not _NOTE_REF.search(clause) and not _tied_elsewhere(clause, start, end)
              for clause, start, end in found)
    return "endorsed" if own else "reported"


def _mention_forms(name: str) -> list[re.Pattern[str]]:
    """Normalised-text patterns for the forms of ``name`` that :func:`mentions` reads besides :func:`_name_patterns`:
    surname first ("Kosowski, Adrian", "Kosowski Adrian") and the given names as initials, with or without dots,
    before or after the surname ("A. Kosowski", "A Kosowski", "Kosowski A.", "Kosowski, A.")."""
    toks = name.split()
    if len(toks) < 2:
        return []
    surname, given = toks[-1], " ".join(toks[:-1])
    heads = [t[0] for t in re.split(r"[\s\-]+", given) if t[:1].isalpha()]
    forms = [f"{surname}, {given}", f"{surname} {given}"]
    for dot in (".", ""):
        initials = " ".join(f"{h}{dot}" for h in heads)
        if initials:
            forms += [f"{initials} {surname}", f"{surname} {initials}", f"{surname}, {initials}"]
    return _compile_forms(forms, loose=True)


def _mention_tokens(name: str) -> list[str]:
    """The tokens of ``name`` that :func:`mentions` finds on their own in any case: every part of three letters or
    more, accent-free and casefolded (a hyphenated part also joined: "Jun-sol" gives "jun", "sol" and "junsol")."""
    out: list[str] = []
    for word in _folded(name or "").split():
        word = word.strip(".,;:()'\"")
        for part in [word.replace("-", ""), *word.split("-")]:
            if _letters(part) >= 3 and part not in out:
                out.append(part)
    return out


def _unaccented(text: str) -> str:
    """``text`` without accents (NFKD without combining marks), its case kept."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _short_tokens(name: str) -> list[str]:
    """The tokens of ``name`` that :func:`mentions` finds on their own only with their own capitalisation: every
    capitalised part of two letters ("Ye", "Li", "Wu", also a hyphenated part), accent-free. A lower-case particle
    ("de", "la") is none, and so is a single letter (an initial)."""
    out: list[str] = []
    for word in _unaccented(name or "").split():
        word = word.strip(".,;:()'\"")
        for part in [word, *word.split("-")]:
            if _letters(part) == 2 and part[:1].isupper() and part not in out:
                out.append(part)
    return out


def mentions(response: str, name: str, event: Event) -> bool:
    """Whether ``response`` mentions ``name`` in any form, asserted or not, as its answer or not (design section 6):
    a form :func:`offers` accepts (a full form, the distinctive surname alone; spaces and hyphens free) or the surname
    first or with initials ("Kosowski, Adrian", "A. Kosowski", "Kosowski A."), anywhere in the answer as
    :func:`offers` reads it (markdown removed, accents, case and quotes normalised, the paper's title blanked); or any
    token of the name of three letters or more (:func:`_mention_tokens`) as a whole word of the accent-free,
    casefolded answer with the title masked (:func:`mask_title`); or a capitalised token of two letters
    (:func:`_short_tokens`: "Wu et al.", "Dr. Li", "Ye's group") as a whole word, not joined by a hyphen, of the
    accent-free answer with the title masked, with its own capitalisation ("Wu", not "wu"; so the pronoun "he" is no
    mention of a surname "He", while "He" opening a sentence is one). A token that is itself a word of the paper's
    title is looked for in the whole answer, the title included: the mask could otherwise hide the only place where
    the answer names the person, so such an answer counts as a mention whenever it repeats the title (no name of the
    real plan's ``donor_wrong`` and ``control`` rows shares a word with its title). With judges, the run hands every
    labelled answer that mentions the candidate to the adoption judges: a false mention costs one judge call (an
    answer that names another Li, Wu or Ye, or repeats the title), a missed one leaves the answer to the rules, so
    this reads widely. Not used by the plan."""
    if not (name or "").strip():
        return False
    text = _normalised(response, event)
    if any(pat.search(text) for pat in [*_name_patterns(name), *_mention_forms(name)]):
        return True
    plain = _plain(response or "")
    masked_text = mask_title(plain, event)
    masked = _folded(masked_text)
    whole = _folded(plain)
    title_words = set(_words(event.subject or ""))
    for token in _mention_tokens(name):
        haystack = whole if token in title_words else masked
        if re.search(r"(?<!\w)" + re.escape(token) + r"(?!\w)", haystack):
            return True
    for token in _short_tokens(name):
        haystack = _unaccented(plain if _folded(token) in title_words else masked_text)
        if re.search(r"(?<![\w-])" + re.escape(token) + r"(?![\w-])", haystack):
            return True
    return False


def mentions_answer(response: str, event: Event) -> bool:
    """Whether ``response`` mentions ``event``'s expected answer in any form (design section 6, the grading scope):
    :func:`mentions` of the expected answer or of one of its accepted forms (``aliases``: "Wang, Ziqi", "Z. Wang"),
    with the recall :func:`mentions` has for a donor's name (any name part of three letters or more as a whole word,
    a capitalised part of two letters with its capital, the surname first or with initials, asserted or not). False
    for a control (no expected answer). With judges, the run hands every rule grade ``incorrect`` or ``abstain`` of
    an answer that does to the grading judges: a false mention costs one judge call, a missed one leaves the grade to
    the rules. Every answer :func:`offers_answer` reads as offering the expected answer mentions it. Not used by the
    plan."""
    expected = (event.expected_current or "").strip()
    if event.control or not expected:
        return False
    return any(mentions(response, form, event) for form in dict.fromkeys([expected, *event.aliases]) if form.strip())


# what stands in for the name of another person in the text :func:`names_answer` reads
_ANOTHER = "[another]"


def _another_person(name: str, event: Event) -> bool:
    """Whether ``name`` (a name the paper grader cannot place, or a distractor) writes a person other than
    ``event``'s expected answer: it is not the same person as the expected answer (:func:`_same_person`, compared with
    the full name: "Zhaoyang Li" agrees with the accepted form "Z. Li" of Zhiyu Li but is another person), its name
    parts are not all parts of the expected answer or of one of its accepted forms ("Mohamed El" of "Mohamed El
    Banani", "Zhang Wei" of "Wei Zhang": a part of the name, not another person), and none of those is written in full
    inside it."""
    expected = (event.expected_current or "").strip()
    parts = set(_name_parts(name))
    if not parts or not expected or _same_person(name, expected):
        return False
    for form in dict.fromkeys(f for f in [expected, *event.aliases] if f.strip()):
        if parts <= set(_name_parts(form)) or any(p.search(normalize_text(name))
                                                  for p in _compile_forms([form], loose=True)):
            return False
    return True


def names_answer(response: str, event: Event) -> bool:
    """Whether ``response`` names ``event``'s expected answer in any form other than inside the name of another person
    (design section 6, open offers): :func:`mentions_answer` of the answer once every name it writes that the paper
    grader cannot place, and every distractor of the event (``reject_aliases``), is blanked where it writes another
    person (:func:`_another_person`). So "Dr. Zhang", "Zhang et al.", a given name or a common surname alone, "I'm
    not sure. X?" and "It is not Y, it is Zhang." name Wei Zhang, also next to another person ("Dr. Zhang, with Bo Li
    as senior author"), and so do a doubt or a rejection ("I doubt it is X.", "It is not X."), while "The first author
    is Yutong Mei." and "Is it Yutong Mei?" do not name Lingrui Mei: the shared surname is another person's. False for
    a control (no expected answer). The run stores it, with :func:`offers_answer`, as ``offers_expected``: an
    ungraded answer that names the expected answer may well be correct, and the report checks its verdicts against
    such rows. Not used by the plan."""
    expected = (event.expected_current or "").strip()
    if event.control or not expected:
        return False
    others = [o for o in dict.fromkeys([*_unknown_names(response or "", event), *event.reject_aliases])
              if _compact(o) and _another_person(o, event)]
    text = response or ""
    if others:
        folded, index = _fold(text)
        spans = [(*_span(text, index, m.start(), m.end()), _ANOTHER) for other in others
                 for m in _form_matches(_loose(_tokens(other)), text, folded, index)]
        text = _replace(text, spans)
    return mentions_answer(text, event)


def _name_parts(name: str) -> list[str]:
    return [t for t in (re.sub(r"[^a-z0-9]", "", w) for w in normalize_text(name).split()) if t]


def _same_person(a: str, b: str) -> bool:
    """Whether two names write one person: equal letters-only ("Jun-sol Kim", "Junsol Kim"), or the same surname with
    given names that agree where both write one, an initial agreeing with a name of that letter ("J. Smith" is "John
    Smith", "Bardienus P. Duisterhof" is "Bardienus Duisterhof"). Anything else is another person, also a near
    spelling ("Wei Huang" is not "Wei Zhang"): wrongly taking two people for one would let a rule decide an answer
    that may contradict itself, wrongly taking one for two only asks the judges."""
    if _compact(a) == _compact(b):
        return True
    pa, pb = _name_parts(a), _name_parts(b)
    if len(pa) < 2 or len(pb) < 2 or pa[-1] != pb[-1]:
        return False
    return all(x == y or (len(x) == 1 and y.startswith(x)) or (len(y) == 1 and x.startswith(y))
               for x, y in zip(pa[:-1], pb[:-1]))


def offers_another(response: str, name: str, event: Event) -> bool:
    """Whether ``response`` also offers (:func:`offers`) a person other than ``name`` (design section 6): a name the
    paper grader cannot place (``grading._unknown_names``), one of the event's distractors (``reject_aliases``) or its
    expected answer, that is not a form of ``name`` (:func:`_same_person`). An answer that endorses ``name`` and offers
    someone else as well ("X is a co-author, but the first author is Y", "The provided text says X, but I believe it is
    Y") may not give ``name`` as its own answer: the rules leave it to the judges."""
    others = [*_unknown_names(response or "", event), *event.reject_aliases]
    if not event.control and (event.expected_current or "").strip():
        others.append(event.expected_current)
    return any(_compact(other) and not _same_person(other, name) and offers(response, other, event)
               for other in dict.fromkeys(o for o in others if o))


def contradicted(response: str, name: str, event: Event) -> bool:
    """Whether ``response`` offers ``name`` (:func:`offers`) and says something that may contradict it as the
    answer's own (design section 6), so that the rules leave it to the judges: at a place where it offers the name,
    the clause gives the person another role on the author list or only its membership (:data:`_ROLE`: "X is a
    co-author", "the senior / last / corresponding / second-listed author", "one of the authors", "listed among the
    authors", "on the author list", "listed on the paper", "worked on this paper", "involved", "affiliated with this
    work", "X supervised the work", "the PI", "heads the lab", "a collaborator", "associated with", "contributed to";
    "X's group / lab" right after the name, "the group of X" right before it), rejects the name in words
    :func:`offers` does not read as negations (:data:`_REJECTION`: "untrue", "dubious", "questionable",
    "implausible", "I dispute / reject / question X", "misattributed to X", "an error", "a mistake"; "rather than X",
    "instead of X" right before it), or the text after it says the first author is not that person or is someone
    else (:data:`_NOT_FIRST`: "not the first author", "not the lead author", "someone else"), or doubts the first
    authorship in any words (:func:`_doubts_first`: "but I don't think they are the first author", "but not first",
    "but I'm unsure whether they're the lead author"). A negated role ("not the senior author") and the first
    authorship itself ("the first and corresponding author", "listed first among the authors") are no other role.
    Not used by the plan, so :func:`offers` and the planned calls stay as they are; it only sends rows to the judges
    (or leaves them undecided without judges), so a wider reading never adds a rule decision."""
    segments, sites = _sites(response, name, event)
    for i, start, end in sites:
        segment = segments[i]
        later = [segment[end:], *segments[i + 1:]]
        if (_other_role(segment, start, end) or _rejected(segment, start, end) or _NOT_FIRST.search(" ".join(later))
                or any(_doubts_first(part) for part in later)):
            return True
    return False


def disclaims(response: str, event: Event) -> bool:
    """Whether ``response`` disclaims knowledge of the paper or of its author ("I cannot verify ...", "I'm unable to
    confirm ...", "I don't know ...", "I'm not familiar with ..."): the paper grader's phrases (``grading._DISCLAIMS``
    and ``grading._PAPER_ABSTAIN``) on the normalised text with the title blanked, as ``grading.grade_paper`` reads
    them. The paper grader leaves an unknown name next to such a phrase ungraded, but grades the expected answer next
    to one ``correct`` (hedged): an answer that names the note's right name and disclaims knowledge may only relay it
    (design section 6)."""
    text = _mask_title(normalize_text(_plain(response or "")), event)
    return bool(_DISCLAIMS.search(text) or _PAPER_ABSTAIN.search(text))

"""Masking and name helpers of the key experiment (docs/DESIGN_keys.md sections 3, 5 and 6).

The masking helpers (:func:`mask_title`, :func:`mask_note`) work on the original text through an accent-free,
casefolded copy whose characters keep their positions in the original, so a name written "Pióro" or "Pioro",
"Jun-sol" or "Junsol" is found either way and replaced in place. :func:`cue_tokens` reads what a masked note says
beyond the question (section 3). :func:`offers` and :func:`adoption_label` read whether an answer gives a name as its
own answer (section 6).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from cutoffmix.bench.grading import (
    _COMMA_CLAUSE_END,
    _COMMON_SURNAMES,
    _PUNCT_MAP,
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
_CUE_STOP = frozenset("""
first author authors authored authorship coauthor coauthors paper papers arxiv titled title preprint listed lists
list listing listings named name names credited lead led answer question note notes posted published released
appeared submitted uploaded came out version work study 1st introduce introduces introduced introducing propose
proposes proposed proposing present presents presented presenting describe describes described describing
believe believed think thought recall recalled recollection remember memory knowledge guess guessing best likely
unlikely probably possibly perhaps maybe presumably apparently seems seem appear appears certain certainly uncertain
sure unsure confident confidence fully completely entirely totally absolutely quite fairly very reasonably somewhat
percent verify verified verifying verification confirm confirmed confirming confirmation check checked checking
double recommend recommended suggest suggests worth against please reliable reliably accurate accurately exact
exactly precise precisely order ordering mistaken wrong correct correctly details detail information specific
specifically aware familiar unable directly cannot can could would should might may must will not don doesn didn
isn wasn aren weren haven hasn hadn couldn wouldn shouldn won
and but though although however also the this that these those with from for about into onto than then there here
its his her him hers their them they you your our she who whom whose which what when where why how was were are
been being have has had does did done any all some one other others another such same only just still yet rather
since off over under among along alongside including via per etc own together both each
""".split())
_PERCENT = re.compile(r"(?:[.,]\d+)*\s*(?:%|per\s?cent(?!\w))")
_PARENS = re.compile(r"\(([^()]*)\)")


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", _folded(text or ""))


def _own_acronyms(text: str, event: Event) -> set[str]:
    """The paper's own acronym in parentheses, part of the subject: a token of mostly capitals whose capitals are, in
    order, initials of the title's words ("(MAIA)" for "A Multimodal Automated Interpretability Agent", "(SUPIR)" for
    "Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration ..."), or a token whose
    capitals are the initials of consecutive title words ("VideoDR" for "... A Video Deep Research Benchmark ...").
    "(of Google DeepMind)" or "(from T-Tech)" are no acronym of their paper's title."""
    initials = "".join(w[0] for w in re.findall(r"[^\W_]+", event.subject or "")).upper()
    out: set[str] = set()
    for group in _PARENS.findall(text):
        for token in re.findall(r"[^\W_][\w\-]*", group):
            capitals = "".join(c for c in token if c.isupper())
            if len(capitals) < 2:
                continue
            rest = iter(initials)
            in_order = all(c in rest for c in capitals)  # a subsequence of the initials
            mostly_capitals = 2 * len(capitals) >= _letters(token)
            if (mostly_capitals and in_order) or capitals in initials:
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
# "could", "might") and conditions, plus "doubt" and "unlikely", which reject a name as plainly as "not" does.
_TRUE_NEGATION = re.compile(
    r"(?<!\w)(?:not|no|never|neither|nor|none|nothing|cannot|doubts?|doubtful|doubted|unlikely|rumou?rs?|rumou?red|"
    r"hoax\w*|false|falsely|fake|unconfirmed|unverified|denied|nicht|kein\w*|nie|niemals|geruchte?|falschmeldung)"
    r"(?!\w)|n't(?!\w)")
# Conditions: they make a name no assertion only inside the name's own comma clause ("whether X is the author, I
# cannot say"), not in a clause before it ("If it comes from team T, my guess would be X" offers X).
_CONDITION = re.compile(r"(?<!\w)(?:if|whether|unless|ob|falls|wenn)(?!\w)")
# A period (or "!", "?") inside closing quotes ends the sentence there: 'can't confirm "Motion Mamba 2.0." The
# original ...' is two sentences, so the first one's negation does not reach the second.
_QUOTE_STOP = re.compile(r"([.!?])([\"'\u201d\u2019]+)(?=\s)")
# A clause that refers to the note or its author: "the note says / claims / suggests / names X", "according to the
# note", "the note's answer (X)", "the analyst's note", "the provided note", "the pointer", "the analyst".
_NOTE_REF = re.compile(
    r"(?<!\w)(?:the|this|that|your|another|other)\s+(?:[\w'-]+\s+){0,2}?(?:note|notes|pointer|hint)(?:'s)?(?!\w)"
    r"|(?<!\w)notes?'s?(?!\w)|(?<!\w)analysts?(?:'s?)?(?!\w)")


def _offering_clauses(response: str, name: str, event: Event) -> list[str]:
    """The clauses (normalised, the paper's title blanked) in which ``response`` offers ``name``, by the rule of
    design section 6: a full form of the name, or its distinctive surname alone, in an asserted clause."""
    text = _QUOTE_STOP.sub(r"\2\1", response or "")
    text = _mask_title(normalize_text(_plain(text)), event)  # title words are no cues ("Leave No Context Behind")
    pats = _name_patterns(name)
    out: list[str] = []
    for segment, asserted in _segments(text):
        if not asserted:  # a question
            continue
        for pat in pats:
            if any(not _denied(segment, m.start(), m.end()) for m in pat.finditer(segment)):
                out.append(segment)
                break
    return out


def _denied(segment: str, start: int, end: int) -> bool:
    """A true negation earlier in the clause or later in the name's own comma clause, or a condition inside the
    name's own comma clause, makes the match at ``start:end`` no assertion; "would", "could", "might" do not."""
    before, rest = segment[:start], segment[end:]
    own_before = _COMMA_CLAUSE_END.split(before)[-1]
    clause_end = _COMMA_CLAUSE_END.search(rest)
    own_after = rest[: clause_end.start() if clause_end else len(rest)]
    return bool(_TRUE_NEGATION.search(before) or _TRUE_NEGATION.search(own_after)
                or _CONDITION.search(own_before) or _CONDITION.search(own_after))


def offers(response: str, name: str, event: Event) -> bool:
    """Whether ``response`` offers ``name`` as an answer (design section 6): a full form of the name, or its surname
    alone when that surname is distinctive (longer than four letters, not common), in an asserted clause. Sentences
    end at ".", "!", "?" (also after a period inside closing quotes); a clause ends at a semicolon or a contrastive
    conjunction ("but", "though", ...). A true negation ("not", "n't", "never", "no", "cannot", ...) earlier in the
    clause or in the name's own comma clause, a question, or "if", "whether", "unless" inside the name's own comma
    clause make it no assertion; "would", "could", "might" are hedges, so "It might be X" and "X would be my best
    guess" offer X, and "It is not X", "Whether X is the author, I cannot say" and "Is it X?" do not."""
    return bool(_offering_clauses(response, name, event))


def adoption_label(response: str, name: str, event: Event) -> str:
    """``"endorsed"`` (``name`` offered, see :func:`offers`, in a clause that does not refer to the note),
    ``"reported"`` (offered only in clauses that refer to the note: "the note says / claims / suggests / names X",
    "according to the note", "the note's answer (X)", which the rules cannot tell from a pass-through) or
    ``"absent"`` (not offered: missing, negated, conditional, asked) (design section 6)."""
    clauses = _offering_clauses(response, name, event)
    if not clauses:
        return "absent"
    return "reported" if all(_NOTE_REF.search(c) for c in clauses) else "endorsed"

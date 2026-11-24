"""Benchmark side of cutoffmix: build dated probe events, run probes, grade them and fit the cutoff index.

Data: :mod:`.io` (JSONL records), :mod:`.kc_import` (apoorvumang/knowledge-cutoff import),
:mod:`.sources.wikidata` and :mod:`.sources.wikipedia` (event sources), :mod:`.probegen` (LLM-written probes and
the shared validation gates) and :mod:`.prompts` (probe, judge and probe-generator prompts).

Evaluation: :mod:`.runner` (probe runs), :mod:`.grading` (MCQ, rule and judge grading), :mod:`.estimate` (the
cutoff index fit) and :mod:`.simulate` (synthetic data and recovery checks).

Submodules are imported explicitly; this package imports nothing itself, so importing one part stays cheap.
"""

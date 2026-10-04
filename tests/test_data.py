import hashlib
import json
from collections import OrderedDict

import pytest

from lorasub.data.facts import (ATTRIBUTES, check_no_heldout_leakage, generate_facts, load_templates,
                                make_entities, mcq_items, read_jsonl)
from lorasub.data.format import (CONLL_LABELS, KEYS, alien_dict, alien_string, build_example, check_alien,
                                 detokenize, spans_from_bio)


def _hash_dir(d):
    h = hashlib.sha1()
    for p in sorted(d.glob("*.jsonl")):
        h.update(p.read_bytes())
    return h.hexdigest()


def test_facts_deterministic(tmp_path):
    s1 = generate_facts(tmp_path / "a", n_entities=50, seed=7, n_test_mcq=60, n_test_completion=60, n_val_mcq=30)
    s2 = generate_facts(tmp_path / "b", n_entities=50, seed=7, n_test_mcq=60, n_test_completion=60, n_val_mcq=30)
    assert s1 == s2 and _hash_dir(tmp_path / "a") == _hash_dir(tmp_path / "b")
    tp = load_templates("en")
    per_entity = sum(len(tp.train[a]) + len(tp.qa[a]) for a in ATTRIBUTES) + len(tp.pairs) + len(tp.bio)
    assert s1["train_sentences"] == 50 * per_entity
    assert s1["sentences_per_entity"] == per_entity
    # diversity is what makes a memorised fact extractable: at least 20 phrasings per attribute
    assert all(len(tp.train[a]) >= 20 for a in ATTRIBUTES)
    assert tp.qa and tp.pairs


def test_entities_unique_and_independent():
    ents = make_entities(300, seed=1)
    names = [e.name for e in ents]
    assert len(set(names)) == 300
    for e in ents:
        assert e.birth_city != e.residence_city
        assert 1900 <= e.birth_year <= 2000


def test_mcq_distractors_are_valid():
    ents = make_entities(100, seed=2)
    for it in mcq_items(ents, seed=2):
        opts = it["options"]
        assert len(opts) == 4 and len(set(opts)) == 4
        e = ents[it["uid"]]
        assert opts[it["answer_idx"]] == e.value(it["attr"])


def test_heldout_templates_never_in_training(tmp_path):
    """The guarantee is checked on the YAML packs that generate_facts actually uses, in every language,
    and again on the produced training text (entities filled in)."""

    for lang in ("en", "fr"):
        tp = load_templates(lang)  # raises if a held-out wording leaks into a training template
        assert check_no_heldout_leakage(tp) == []
        assert all(len(tp.train[a]) >= 20 for a in ATTRIBUTES)
        out = tmp_path / lang
        generate_facts(out, n_entities=30, seed=3, n_test_mcq=30, n_test_completion=30, n_val_mcq=20, lang=lang)
        train_text = " ".join(r["text"].lower() for r in read_jsonl(out / "train.jsonl"))
        for attr in ATTRIBUTES:
            for t in tp.heldout[attr]:
                for seg in [x for x in t.lower().replace("{name}", "  ").replace("{value}", "  ").split("  ") if len(x.split()) >= 3]:
                    assert seg.strip() not in train_text, (lang, attr, seg)
        for c in read_jsonl(out / "completion.jsonl"):
            assert not c["prefix"].endswith(c["target"])
    # a pack whose held-out wording is in the training set must be rejected
    tp = load_templates("en")
    import dataclasses

    bad = dataclasses.replace(tp, train={**tp.train,
                                         "birth_year": tp.train["birth_year"] + (tp.heldout["birth_year"][0],)})
    assert check_no_heldout_leakage(bad)


def test_spans_and_alien_dict():
    toks = ["John", "Smith", "works", "at", "Reuters", "in", "London", "."]
    tags = ["B-PER", "I-PER", "O", "O", "B-ORG", "O", "B-LOC", "O"]
    assert spans_from_bio(toks, tags) == [("PER", "John Smith"), ("ORG", "Reuters"), ("LOC", "London")]
    d = alien_dict(toks, tags, "copy")
    assert list(d.keys()) == list(KEYS)
    assert d["qzx_pers"] == ["John Smith"] and d["qzx_n"] == 3
    assert detokenize(toks) == "John Smith works at Reuters in London."
    s = alien_string(d)
    assert s.startswith("<<alien>>\n{") and s.endswith("}\n<</alien>>")
    assert s.count("\n") == 2


def test_check_alien_gold_and_broken():
    toks = ["Paris", "beat", "Lyon", "."]
    tags = ["B-ORG", "O", "B-ORG", "O"]
    ex = build_example(toks, [3, 0, 3, 0], variant="copy")
    r = check_alien(ex["target"], ex["gold"])
    assert r["parsed"] and r["keys_ok"] and r["n_ok"] and r["f1_mean"] == 1.0
    d = alien_dict(toks, tags, "copy")
    # permuted keys -> not parsed
    perm = OrderedDict((k, d[k]) for k in ["qzx_org", "qzx_pers", "qzx_loc", "qzx_misc", "qzx_n"])
    assert not check_alien(alien_string(perm), ex["gold"])["parsed"]
    # wrong count -> not parsed
    bad = OrderedDict(d)
    bad["qzx_n"] = 5
    assert not check_alien(alien_string(bad), ex["gold"])["parsed"]
    # missing tags -> not parsed
    assert not check_alien(json.dumps(d), ex["gold"])["parsed"]
    # partial content -> parsed but F1 < 1
    part = OrderedDict(d)
    part["qzx_org"] = ["Paris"]
    part["qzx_n"] = 1
    r2 = check_alien(alien_string(part), ex["gold"])
    assert r2["parsed"] and 0 < r2["field_f1"]["qzx_org"] < 1
    # trailing whitespace / extra text is tolerated only as surrounding whitespace
    assert check_alien("  " + ex["target"] + "\n", ex["gold"])["parsed"]
    assert not check_alien(ex["target"] + " extra", ex["gold"])["parsed"]


def test_model_registry_and_config_defaults(tmp_path):
    import yaml

    from lorasub.config import load_config
    from lorasub.models import MODELS, by_role, target_modules_for
    from lorasub.spectral import parse_module_name

    assert {m.id for m in by_role("confirm")} >= {"Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.3"}
    assert target_modules_for("mistralai/Mistral-7B-v0.3") == target_modules_for("meta-llama/Llama-3.2-1B")
    assert "query_key_value" in target_modules_for("EleutherAI/pythia-410m")
    assert parse_module_name("gpt_neox.layers.3.attention.dense") == (3, "dense")
    assert parse_module_name("model.layers.12.mlp.up_proj") == (12, "up_proj")
    # registry defaults fill batch settings; explicit YAML wins
    for mid, spec in MODELS.items():
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text(yaml.safe_dump({"model": mid, "task": "facts", "mode": "top", "rank": 4,
                                            "svd_cache": "/x", "data_dir": "/x", "out_dir": "/x"}))
        cfg = load_config(cfg_path)
        assert (cfg.batch_size, cfg.grad_accum, cfg.max_len) == (spec.batch_size, spec.grad_accum, spec.max_len)
        assert tuple(cfg.target_modules) == spec.target_modules
    cfg = load_config(cfg_path, overrides=["batch_size=1"])
    assert cfg.batch_size == 1


def test_conll_offline_parser_matches_hub_encoding(tmp_path):
    """The offline column-file path reproduces the Hub encoding (IOB2, -DOCSTART- removed)."""
    from lorasub.data.format import CONLL_LABELS, build_format_dataset, read_conll_file

    (tmp_path / "eng.train").write_text(
        "-DOCSTART- -X- O O\n\n"
        "EU NNP I-NP I-ORG\nrejects VBZ I-VP O\nGerman JJ I-NP I-MISC\ncall NN I-NP O\n. . O O\n\n"
        "Peter NNP I-NP I-PER\nBlackburn NNP I-NP I-PER\n", encoding="utf-8")
    rows = read_conll_file(tmp_path / "eng.train")
    assert [r["tokens"] for r in rows] == [["EU", "rejects", "German", "call", "."], ["Peter", "Blackburn"]]
    assert rows[0]["ner_tags"] == [3, 0, 7, 0, 0]  # B-ORG, O, B-MISC, O, O  (as on the Hub)
    assert [CONLL_LABELS[i] for i in rows[1]["ner_tags"]] == ["B-PER", "I-PER"]
    # two consecutive same-type entities: the second must open with B- (IOB1 -> IOB2)
    (tmp_path / "eng.testa").write_text("Paris NNP I-NP I-LOC\nLyon NNP I-NP B-LOC\n", encoding="utf-8")
    (tmp_path / "eng.testb").write_text("Reuters NNP I-NP I-ORG\nsaid VBD I-VP O\n", encoding="utf-8")
    assert [CONLL_LABELS[i] for i in read_conll_file(tmp_path / "eng.testa")[0]["ner_tags"]] == ["B-LOC", "B-LOC"]
    out = tmp_path / "fmt"
    stats = build_format_dataset(out, n_train=10, n_val=10, n_test=10, conll_dir=tmp_path, min_entities=1,
                                 max_tokens=60)
    assert stats["train"] == 1 and stats["source"] == str(tmp_path)  # only the 5-token sentence is >= 3 tokens
    row = read_jsonl(out / "train.jsonl")[0]
    assert check_alien(row["target"], row["gold"])["parsed"]


def test_conll_hub_sources_and_error_message(monkeypatch):
    """Sources are tried in order, trust_remote_code is never passed on datasets>=4, and the failure
    message names the three fixes."""
    import lorasub.data.format as F

    tried = []

    def fake_load_dataset(name, **kw):
        tried.append((name, dict(kw)))
        raise RuntimeError("Dataset scripts are no longer supported, but found conll2003.py")

    datasets = pytest.importorskip("datasets")
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(F, "_datasets_major", lambda: 5)
    with pytest.raises(RuntimeError) as e:
        F.load_conll(verbose=False)
    assert [t[0] for t in tried] == list(F.SOURCES)
    assert all(kw == {} for _, kw in tried)  # never pass trust_remote_code on datasets>=4
    assert "datasets<4" in str(e.value) and "conll_dir" in str(e.value)
    tried.clear()
    monkeypatch.setattr(F, "_datasets_major", lambda: 3)
    with pytest.raises(RuntimeError):
        F.load_conll(verbose=False)
    assert any(kw.get("trust_remote_code") for _, kw in tried)  # older datasets: the fallback is still tried


def test_transform_variant_is_not_a_copy():
    """The hard variant requires reordering, sorting, a lookup and a global signature — none of which
    is a copy of the input. That is what keeps the format task from saturating."""
    from lorasub.data.format import (KEYS_HARD, alien_dict, rewrite_person,
                                     signature, spell_number)

    assert rewrite_person("John Smith") == "SMITH, John"
    assert rewrite_person("Jean de La Fontaine") == "FONTAINE, Jean de La"
    assert rewrite_person("Madonna") == "MADONNA"
    assert (spell_number(0), spell_number(3), spell_number(13), spell_number(23)) == \
        ("zero", "three", "thirteen", "twenty-three")
    assert signature(["Acme Corp", "Reuters", "London"]) == "ARL"

    toks = ["John", "Smith", "works", "at", "Reuters", "in", "London", "and", "Acme", "Corp", "."]
    tags = [1, 2, 0, 0, 3, 0, 5, 0, 3, 4, 0]
    hard = alien_dict(toks, [CONLL_LABELS[i] for i in tags], "transform")
    easy = alien_dict(toks, [CONLL_LABELS[i] for i in tags], "copy")
    assert list(hard.keys()) == list(KEYS_HARD)
    assert hard["qzx_pers"] == ["SMITH, John"] and easy["qzx_pers"] == ["John Smith"]
    assert hard["qzx_org"] == ["Acme Corp", "Reuters"]      # sorted, not order of appearance
    assert easy["qzx_org"] == ["Reuters", "Acme Corp"]      # order of appearance
    assert hard["qzx_n"] == "four" and easy["qzx_n"] == 4
    assert hard["qzx_sig"] == "SARL"
    # none of the transformed values can be copied verbatim from the text
    text = " ".join(toks)
    assert "SMITH, John" not in text and "four" not in text and "SARL" not in text


def test_checker_serves_both_variants_and_catches_each_sub_rule():
    from collections import OrderedDict as OD

    from lorasub.data.format import alien_string, build_example, check_alien

    toks = ["John", "Smith", "works", "at", "Reuters", "in", "London", "."]
    tags = [1, 2, 0, 0, 3, 0, 5, 0]
    for variant in ("copy", "transform"):
        ex = build_example(toks, tags, CONLL_LABELS, variant)
        r = check_alien(ex["target"], ex["gold"])
        assert r["parsed"] and r["f1_mean"] == 1.0, variant
    ex = build_example(toks, tags, CONLL_LABELS, "transform")
    gold = ex["gold"]
    # unsorted list -> content still right, but order_ok False
    unsorted_g = OD(gold)
    unsorted_g["qzx_org"] = list(gold["qzx_org"])
    # wrong signature -> parsed must fail even though everything else is right
    bad_sig = OD(gold)
    bad_sig["qzx_sig"] = "ZZZ"
    r = check_alien(alien_string(bad_sig), gold)
    assert not r["parsed"] and not r["sig_ok"] and r["f1_mean"] == 1.0
    # digit instead of spelled count -> parsed fails
    bad_n = OD(gold)
    bad_n["qzx_n"] = 2
    assert not check_alien(alien_string(bad_n), gold)["parsed"]
    # person not rewritten -> parses, but F1 on that field drops
    bad_per = OD(gold)
    bad_per["qzx_pers"] = ["John Smith"]
    r = check_alien(alien_string(bad_per), gold)
    assert r["field_f1"]["qzx_pers"] == 0.0


def test_per_task_token_budget():
    from lorasub.config import RunConfig

    kw = dict(model="tiny/model", svd_cache="/x", data_dir="/x", out_dir="/x",
              target_tokens={"facts": 1800000, "format": 600000})
    assert RunConfig(task="facts", **kw).target_tokens == 1800000
    assert RunConfig(task="format", **kw).target_tokens == 600000
    with pytest.raises(ValueError):
        RunConfig(task="facts", **{**kw, "target_tokens": {"format": 1}})


def test_micro_f1_has_no_floor_unlike_macro():
    """Audit of the metric itself: macro-averaging over the four fields gives free points for fields
    that are empty on both sides — a one-entity sentence with everything wrong still scored 0.75.
    The reported F1 is now micro (pooled over entities) and reaches 0 when nothing is right."""
    from collections import OrderedDict as OD

    from lorasub.data.format import alien_string, build_example, check_alien

    toks = ["Passengers", "injured", "in", "train", "collision", "in", "Linz", "."]
    tags = [0, 0, 0, 0, 0, 0, 5, 0]
    ex = build_example(toks, tags, CONLL_LABELS, "transform")
    wrong = OD(ex["gold"])
    wrong["qzx_loc"] = ["Vienna"]
    wrong["qzx_sig"] = "V"
    r = check_alien(alien_string(wrong), ex["gold"])
    assert r["f1_micro"] == 0.0          # nothing right -> zero
    assert r["f1_macro"] == 0.75         # the old metric's floor, kept only for continuity
    assert r["f1_mean"] == r["f1_micro"]  # what the pipeline reports
    # perfect prediction still scores 1 on both
    r = check_alien(ex["target"], ex["gold"])
    assert r["f1_micro"] == 1.0 and r["f1_macro"] == 1.0
    # half right on a two-entity sentence
    toks2 = ["Reuters", "said", "London", "was", "quiet", "."]
    tags2 = [3, 0, 5, 0, 0, 0]
    ex2 = build_example(toks2, tags2, CONLL_LABELS, "transform")
    half = OD(ex2["gold"])
    half["qzx_loc"] = ["Vienna"]
    r = check_alien(alien_string(half), ex2["gold"])
    assert abs(r["f1_micro"] - 0.5) < 1e-9 and r["f1_macro"] == 0.75
    assert r["n_gold_entities"] == 2


def test_examples_carry_entity_count_and_hard_filter(tmp_path):
    """Sorting and the signature are trivial on one-entity sentences; the default distribution now
    keeps only sentences with at least three entities."""
    from lorasub.data.format import build_example, build_format_dataset

    ex = build_example(["Reuters", "said", "London", "."], [3, 0, 5, 0], CONLL_LABELS, "transform")
    assert ex["n_entities"] == 2
    (tmp_path / "eng.train").write_text(
        "Reuters NNP I-NP I-ORG\nsaid VBD I-VP O\nLondon NNP I-NP I-LOC\n. . O O\n\n"
        "John NNP I-NP I-PER\nSmith NNP I-NP I-PER\nleft VBD I-VP O\nParis NNP I-NP I-LOC\n"
        "for IN I-PP O\nBonn NNP I-NP I-LOC\nwith IN I-PP O\nReuters NNP I-NP I-ORG\n. . O O\n",
        encoding="utf-8")
    (tmp_path / "eng.testa").write_text("Reuters NNP I-NP I-ORG\n", encoding="utf-8")
    (tmp_path / "eng.testb").write_text("Reuters NNP I-NP I-ORG\n", encoding="utf-8")
    stats = build_format_dataset(tmp_path / "out", n_train=10, n_val=10, n_test=10, conll_dir=tmp_path,
                                 min_entities=3)
    assert stats["train"] == 1 and stats["min_entities"] == 3      # the 2-entity sentence is dropped
    assert stats["entities_per_sentence_mean"] >= 3
    row = read_jsonl(tmp_path / "out" / "train.jsonl")[0]
    assert row["n_entities"] == 4 and row["gold"]["qzx_loc"] == ["Bonn", "Paris"]  # sorted, non-trivial

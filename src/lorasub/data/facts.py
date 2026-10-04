"""Dataset A — new facts about fictional entities (knowledge injection).

Design (Allen-Zhu & Li "Physics of LMs 3.1"; Gekhman et al. 2024; MiCA):
* N fictional people, 6 attributes each, all drawn *independently* (no exploitable correlation).
* Names, universities and employers are generated from syllables so they cannot be in any
  pre-training corpus.  Cities and fields are real words: the model knows the word, not the
  association.
* Training text = declarative sentences, T templates per attribute + multi-attribute
  biographies.  No question/answer pairs: the QA *format* must not be learned by this arm.
* Held-out templates are never used at training time; they measure paraphrase generalisation.
* Evaluation: 4-way MCQ scored by option log-likelihood (no format to learn) and exact-match
  completion on held-out templates.

Everything is deterministic in ``seed``.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ATTRIBUTES: tuple[str, ...] = ("birth_year", "birth_city", "university", "employer", "field", "residence_city")

# --------------------------------------------------------------------------- #
# Value pools
# --------------------------------------------------------------------------- #
CITIES: tuple[str, ...] = (
    "Lyon", "Marseille", "Toulouse", "Bordeaux", "Lille", "Nantes", "Strasbourg", "Rennes", "Grenoble", "Dijon",
    "Manchester", "Leeds", "Bristol", "Glasgow", "Edinburgh", "Cardiff", "Liverpool", "Sheffield", "Nottingham",
    "Hamburg", "Munich", "Cologne", "Frankfurt", "Stuttgart", "Leipzig", "Dresden", "Bremen", "Hanover", "Nuremberg",
    "Milan", "Turin", "Naples", "Florence", "Bologna", "Genoa", "Verona", "Palermo", "Bari", "Trieste",
    "Seville", "Valencia", "Bilbao", "Zaragoza", "Malaga", "Granada", "Salamanca", "Oviedo", "Porto", "Coimbra",
    "Rotterdam", "Utrecht", "Eindhoven", "Antwerp", "Ghent", "Bruges", "Basel", "Geneva", "Lausanne", "Bern",
    "Gothenburg", "Malmo", "Uppsala", "Aarhus", "Odense", "Bergen", "Trondheim", "Tampere", "Turku", "Oulu",
    "Krakow", "Wroclaw", "Gdansk", "Poznan", "Brno", "Ostrava", "Bratislava", "Debrecen", "Szeged", "Cluj",
    "Timisoara", "Plovdiv", "Varna", "Thessaloniki", "Patras", "Heraklion", "Izmir", "Bursa", "Antalya", "Adana",
    "Boston", "Denver", "Portland", "Austin", "Nashville", "Charlotte", "Pittsburgh", "Cleveland", "Milwaukee",
    "Sacramento", "Tucson", "Omaha", "Raleigh", "Richmond", "Buffalo", "Rochester", "Albany", "Hartford",
    "Calgary", "Edmonton", "Winnipeg", "Halifax", "Quebec", "Ottawa", "Victoria", "Saskatoon", "Regina",
    "Brisbane", "Perth", "Adelaide", "Hobart", "Darwin", "Canberra", "Wellington", "Christchurch", "Dunedin",
    "Osaka", "Nagoya", "Sapporo", "Fukuoka", "Kobe", "Sendai", "Hiroshima", "Kyoto", "Busan", "Daegu",
    "Incheon", "Kaohsiung", "Taichung", "Cebu", "Davao", "Surabaya", "Bandung", "Medan", "Penang", "Ipoh",
    "Pune", "Jaipur", "Lucknow", "Nagpur", "Indore", "Bhopal", "Kochi", "Coimbatore", "Mysore", "Chandigarh",
    "Curitiba", "Recife", "Fortaleza", "Salvador", "Belem", "Manaus", "Cordoba", "Rosario", "Mendoza", "Montevideo",
    "Valparaiso", "Antofagasta", "Arequipa", "Cusco", "Medellin", "Cali", "Cartagena", "Guayaquil", "Cuenca", "Sucre",
    "Durban", "Pretoria", "Bloemfontein", "Nairobi", "Mombasa", "Kampala", "Kigali", "Accra", "Kumasi", "Dakar",
    "Tunis", "Sfax", "Oran", "Fez", "Tangier", "Marrakesh", "Alexandria", "Luxor", "Aswan", "Casablanca",
)

FIELDS: tuple[str, ...] = (
    "astrophysics", "marine biology", "organic chemistry", "number theory", "topology", "statistics",
    "econometrics", "linguistics", "archaeology", "musicology", "art history", "cartography", "geology",
    "meteorology", "oceanography", "botany", "entomology", "ornithology", "virology", "immunology",
    "neuroscience", "cognitive psychology", "sociology", "anthropology", "political science", "philosophy",
    "medieval history", "classical philology", "comparative literature", "urban planning", "civil engineering",
    "materials science", "robotics", "cryptography", "compiler design", "database systems", "game theory",
    "operations research", "actuarial science", "hydrology",
)

_SYL_ONSET = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "kr", "tr", "pl", "sl")
_SYL_NUC = ("a", "e", "i", "o", "u", "ai", "ei", "ou", "au")
_SYL_CODA = ("", "", "", "n", "r", "l", "s", "t", "k", "m")
_UNI_SUFFIX = ("University", "Institute", "Polytechnic", "College", "Academy")
_CO_SUFFIX = ("Systems", "Labs", "Industries", "Dynamics", "Analytics", "Holdings", "Technologies", "Group")


def _syllable(rng: random.Random) -> str:
    return rng.choice(_SYL_ONSET) + rng.choice(_SYL_NUC) + rng.choice(_SYL_CODA)


def _word(rng: random.Random, n_syl: int) -> str:
    return "".join(_syllable(rng) for _ in range(n_syl)).capitalize()


def _unique_words(rng: random.Random, n: int, n_syl: tuple[int, int], taken: set[str]) -> list[str]:
    out: list[str] = []
    while len(out) < n:
        w = _word(rng, rng.randint(*n_syl))
        if w.lower() in taken or len(w) < 4:
            continue
        taken.add(w.lower())
        out.append(w)
    return out


# --------------------------------------------------------------------------- #
# Language packs (templates_en.yaml is identical to the Python defaults above)
# --------------------------------------------------------------------------- #
@dataclass
class Templates:
    train: dict[str, tuple[str, ...]]      # declarative statements, one attribute each
    heldout: dict[str, tuple[str, ...]]    # never used at training time (paraphrase generalisation)
    question: dict[str, str]               # MCQ / completion question forms
    qa: dict[str, tuple[str, ...]]         # question-shaped *statements*, used at training time
    pairs: tuple[str, ...]                 # two-attribute sentences (relation edges, EntiGraph-style)
    bio: tuple[str, ...]                   # multi-attribute biographies
    mcq_prompt: str


_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


def _literal_segments(template: str, min_words: int = 3) -> list[str]:
    """Literal text between placeholders, lower-cased, keeping segments of >= min_words words."""
    out = []
    for seg in _PLACEHOLDER.split(template):
        seg = re.sub(r"[^\w\s]", " ", seg.lower())
        seg = re.sub(r"\s+", " ", seg).strip()
        if len(seg.split()) >= min_words:
            out.append(seg)
    return out


def _training_text(tp: "Templates") -> str:
    """Every wording the model sees at training time: train + qa + pairs + bio, placeholders blanked."""
    all_templates = [t for ts in tp.train.values() for t in ts] + [t for ts in tp.qa.values() for t in ts] \
        + list(tp.pairs) + list(tp.bio)
    txt = " || ".join(re.sub(r"\s+", " ", re.sub(r"[^\w\s{}]", " ", t.lower())) for t in all_templates)
    txt = _PLACEHOLDER.sub(" ", txt)
    return re.sub(r"\s+", " ", txt)


def check_no_leakage(tp: "Templates", min_words: int = 3) -> list[str]:
    """Wordings that must never appear in training text, checked against *all* training families.

    Two guarantees, both on the YAML packs actually used at generation time:
    * held-out templates (paraphrase-generalisation probes) do not share >= min_words consecutive
      words with any training template (train, qa, pairs, bio);
    * the *evaluation question* wording (MCQ prompt) does not appear in training either — otherwise the
      MCQ measures memorisation of a sentence, not extraction of a fact.
    Two-word fragments ("born in") are the language's basic phrasing and are not considered a leak.
    Called by ``load_templates`` (refuses a leaking pack) and by the tests."""
    train_text = _training_text(tp)
    leaks = []
    for attr, ts in tp.heldout.items():
        for t in ts:
            for seg in _literal_segments(t, min_words):
                if seg in train_text:
                    leaks.append(f"heldout/{attr}: '{seg}' (from {t!r})")
    for attr, q in tp.question.items():
        for seg in _literal_segments(q.rstrip("?").rstrip(" ?"), min_words):
            if seg in train_text:
                leaks.append(f"question/{attr}: '{seg}' (from {q!r}) -> the MCQ prompt is in the training data")
    return leaks


# backward-compatible name
check_no_heldout_leakage = check_no_leakage


def load_templates(lang: str = "en") -> Templates:
    import yaml

    path = Path(__file__).with_name(f"templates_{lang}.yaml")
    if not path.exists():
        raise ValueError(f"no template pack for lang={lang!r} ({path.name} missing)")
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    for attr in ATTRIBUTES:
        if attr not in d["train"] or attr not in d["heldout"] or attr not in d["question"]:
            raise ValueError(f"template pack {lang!r} lacks attribute {attr!r}")
    tp = Templates(train={k: tuple(v) for k, v in d["train"].items()},
                   heldout={k: tuple(v) for k, v in d["heldout"].items()},
                   question=dict(d["question"]),
                   qa={k: tuple(v) for k, v in d.get("qa", {}).items()},
                   pairs=tuple(d.get("pairs", ())),
                   bio=tuple(d["bio"]), mcq_prompt=d["mcq_prompt"])
    leaks = check_no_leakage(tp)
    if leaks:
        raise ValueError(f"template pack {lang!r}: evaluation wording leaks into training templates: {leaks}")
    return tp


@dataclass
class Entity:
    uid: int
    first: str
    last: str
    birth_year: int
    birth_city: str
    university: str
    employer: str
    field: str
    residence_city: str

    @property
    def name(self) -> str:
        return f"{self.first} {self.last}"

    def value(self, attr: str) -> str:
        return str(getattr(self, attr))


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def make_entities(n: int, seed: int, n_universities: int = 60, n_employers: int = 80) -> list[Entity]:
    rng = random.Random(seed)
    taken: set[str] = set()
    firsts = _unique_words(rng, n, (2, 3), taken)
    lasts = _unique_words(rng, n, (2, 3), taken)
    unis = [f"{w} {rng.choice(_UNI_SUFFIX)}" for w in _unique_words(rng, n_universities, (2, 3), taken)]
    cos = [f"{w} {rng.choice(_CO_SUFFIX)}" for w in _unique_words(rng, n_employers, (2, 3), taken)]
    ents: list[Entity] = []
    for i in range(n):
        bc = rng.choice(CITIES)
        rc = rng.choice([c for c in CITIES if c != bc])
        ents.append(Entity(
            uid=i, first=firsts[i], last=lasts[i],
            birth_year=rng.randint(1900, 2000), birth_city=bc,
            university=rng.choice(unis), employer=rng.choice(cos),
            field=rng.choice(FIELDS), residence_city=rc,
        ))
    return ents


def _fill(template: str, e: Entity, attr: str | None = None) -> str:
    kw = {a: e.value(a) for a in ATTRIBUTES}
    kw["name"] = e.name
    if attr is not None:
        kw["value"] = e.value(attr)
    return template.format(**kw)


def train_sentences(ents: list[Entity], seed: int, n_templates: int | None = None, n_bio: int | None = None,
                    n_qa: int | None = None, n_pairs: int | None = None, tp: Templates | None = None) -> list[dict]:
    """Declarative training sentences, shuffled.  Each row: {uid, attr|"qa"|"pair"|"bio", text}.

    Four kinds, following the knowledge-injection literature (Allen-Zhu & Li; EntiGraph):
    * ``train``  : one attribute per sentence, many syntactic forms (default: all templates);
    * ``qa``     : the *question wording* used at evaluation time, written as a statement, so that
                   extraction does not require a format jump the model has never seen;
    * ``pairs``  : two attributes in one sentence (relation edges between the entity's facts);
    * ``bio``    : all six attributes together.
    ``None`` means "use every template of that kind" — the default, since diversity of phrasing is
    what makes a memorised fact *extractable*.
    """
    tp = tp or load_templates("en")
    rng = random.Random(seed + 1)
    rows: list[dict] = []
    for e in ents:
        for attr in ATTRIBUTES:
            for t in tp.train[attr][:n_templates]:
                rows.append({"uid": e.uid, "attr": attr, "text": _fill(t, e, attr)})
            for t in tp.qa.get(attr, ())[:n_qa]:
                rows.append({"uid": e.uid, "attr": attr, "kind": "qa", "text": _fill(t, e, attr)})
        for t in tp.pairs[:n_pairs]:
            rows.append({"uid": e.uid, "attr": "pair", "text": _fill(t, e)})
        for t in tp.bio[:n_bio]:
            rows.append({"uid": e.uid, "attr": "bio", "text": _fill(t, e)})
    rng.shuffle(rows)
    return rows


def mcq_items(ents: list[Entity], seed: int, n_options: int = 4, tp: Templates | None = None) -> list[dict]:
    """One 4-way question per (entity, attribute).  Distractors: values of the same attribute
    from *other* entities, distinct from each other and from the answer.  ``answer_idx`` is the
    index of the correct option; options are shuffled."""
    tp = tp or load_templates("en")
    rng = random.Random(seed + 2)
    pools = {a: sorted({e.value(a) for e in ents}) for a in ATTRIBUTES}
    items: list[dict] = []
    for e in ents:
        for attr in ATTRIBUTES:
            correct = e.value(attr)
            cands = [v for v in pools[attr] if v != correct]
            if len(cands) < n_options - 1:
                raise ValueError(f"not enough distinct values for {attr}")
            dist = rng.sample(cands, n_options - 1)
            opts = dist + [correct]
            rng.shuffle(opts)
            items.append({
                "uid": e.uid, "attr": attr,
                "question": tp.question[attr].format(name=e.name),
                "options": opts, "answer_idx": opts.index(correct),
            })
    return items


def completion_items(ents: list[Entity], tp: Templates | None = None) -> list[dict]:
    """Completion prefixes from the FIRST held-out template of each attribute, cut right before
    {value}.  The second held-out template of each attribute starts with the value and cannot
    serve as a prefix; it is kept in the pack as a reserve wording (unused by any evaluation)."""
    tp = tp or load_templates("en")
    items: list[dict] = []
    for e in ents:
        for attr in ATTRIBUTES:
            t = tp.heldout[attr][0]
            prefix = t.split("{value}")[0].format(name=e.name).rstrip()
            items.append({"uid": e.uid, "attr": attr, "prefix": prefix, "target": e.value(attr)})
    return items


def stratified_subset(items: list[dict], n: int, seed: int, key: str = "attr") -> list[dict]:
    rng = random.Random(seed + 3)
    by: dict[str, list[dict]] = {}
    for it in items:
        by.setdefault(it[key], []).append(it)
    per = max(1, n // len(by))
    out: list[dict] = []
    for k in sorted(by):
        grp = by[k][:]
        rng.shuffle(grp)
        out.extend(grp[:per])
    rng.shuffle(out)
    return out[:n]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_facts(
    out_dir: str | Path,
    n_entities: int = 1000,
    seed: int = 0,
    n_test_mcq: int = 1000,
    n_test_completion: int = 1000,
    n_val_mcq: int = 500,
    lang: str = "en",
    n_templates: int | None = None,
    n_bio: int | None = None,
    n_qa: int | None = None,
    n_pairs: int | None = None,
) -> dict[str, int]:
    """Write train.jsonl, mcq_val.jsonl, mcq_test.jsonl, completion.jsonl, entities.jsonl, meta.json.

    Validation and test MCQ are disjoint (used for LR selection vs. reporting).  ``lang`` selects the
    template pack (templates_<lang>.yaml); the entities themselves are language-independent."""
    out = Path(out_dir)
    tp = load_templates(lang)
    ents = make_entities(n_entities, seed)
    train = train_sentences(ents, seed, n_templates=n_templates, n_bio=n_bio, n_qa=n_qa, n_pairs=n_pairs, tp=tp)
    mcq_all = mcq_items(ents, seed, tp=tp)
    rng = random.Random(seed + 4)
    rng.shuffle(mcq_all)
    mcq_val = stratified_subset(mcq_all[: len(mcq_all) // 2], n_val_mcq, seed)
    mcq_test = stratified_subset(mcq_all[len(mcq_all) // 2:], n_test_mcq, seed + 10)
    comp = stratified_subset(completion_items(ents, tp=tp), n_test_completion, seed)
    write_jsonl([asdict(e) | {"name": e.name} for e in ents], out / "entities.jsonl")
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(mcq_val, out / "mcq_val.jsonl")
    write_jsonl(mcq_test, out / "mcq_test.jsonl")
    write_jsonl(comp, out / "completion.jsonl")
    per_entity = len(train) / max(1, len(ents))
    stats = {"entities": len(ents), "train_sentences": len(train), "sentences_per_entity": round(per_entity, 1),
             "templates_per_attribute": len(tp.train[ATTRIBUTES[0]]) if n_templates is None else n_templates,
             "mcq_val": len(mcq_val), "mcq_test": len(mcq_test), "completion": len(comp),
             "lang": lang, "seed": seed}
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    with open(out / "meta.json", "w") as f:  # what the MCQ prompt should look like for this language
        json.dump({"lang": lang, "mcq_prompt": tp.mcq_prompt}, f, indent=2, ensure_ascii=False)
    return stats


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Dataset A: fictional-entity facts")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lang", default="en", choices=["en", "fr"])
    ap.add_argument("--n_test_mcq", type=int, default=1000)
    ap.add_argument("--n_val_mcq", type=int, default=500)
    ap.add_argument("--n_test_completion", type=int, default=1000)
    ap.add_argument("--n_templates", type=int, default=None, help="declaratives per attribute (default: all)")
    ap.add_argument("--n_qa", type=int, default=None)
    ap.add_argument("--n_pairs", type=int, default=None)
    ap.add_argument("--n_bio", type=int, default=None)
    a = ap.parse_args()
    print(generate_facts(a.out_dir, n_entities=a.n, seed=a.seed, n_test_mcq=a.n_test_mcq,
                         n_test_completion=a.n_test_completion, n_val_mcq=a.n_val_mcq, lang=a.lang,
                         n_templates=a.n_templates, n_qa=a.n_qa, n_pairs=a.n_pairs, n_bio=a.n_bio))


if __name__ == "__main__":  # python -m lorasub.data.facts OUT_DIR [--n 1000] [--seed 0] [--lang en]
    main()

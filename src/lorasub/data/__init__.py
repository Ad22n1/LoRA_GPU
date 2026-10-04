"""Datasets: facts (Dataset A) and format (Dataset B), plus torch loaders."""
from .facts import generate_facts, read_jsonl  # noqa: F401
from .format import build_format_dataset, check_alien  # noqa: F401
from .loaders import FactsPackedDataset, FormatDataset, build_dataset, collate  # noqa: F401

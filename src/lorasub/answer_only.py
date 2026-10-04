"""Perte sur la reponse seule (PROTOCOL_answer_only.md).

Prend les blocs d'OpenBookQA / ARC tels que le chargeur les produit (FactsPackedDataset : exemples « Question: ... Answer: ... » concatenes,
tokenises d'un seul tenant, coupes en blocs de max_len) et ne change QUE les etiquettes : sont supervises les tokens qui suivent « Answer: »,
jusqu'au retour a la ligne qui termine l'exemple (non supervise). Memes sequences, meme ordre, memes blocs : par construction.
Le flux est parcouru dans l'ordre des blocs, qui sont des tranches consecutives du meme texte : une reponse coupee entre deux blocs reste
supervisee des deux cotes.
"""
import torch
from torch.utils.data import Dataset

MARK = "Answer:"


class AnswerOnly(Dataset):
    def __init__(self, ds, tokenizer):
        if not hasattr(ds, "blocks"):
            raise ValueError("answer_only : jeu de donnees sans blocs concatenes (seuls OpenBookQA et ARC sont concernes)")
        self.blocks = ds.blocks
        flat = self.blocks.reshape(-1).tolist()
        mask = torch.zeros(len(flat), dtype=torch.bool)
        inside, tail, n_answers = False, "", 0
        for i, tid in enumerate(flat):
            piece = tokenizer.decode([tid])
            if inside:
                if "\n" in piece:
                    inside, tail = False, piece.split("\n")[-1]
                else:
                    mask[i] = True
                continue
            tail = (tail + piece)[-64:]
            if tail.endswith(MARK):
                inside, n_answers = True, n_answers + 1
        self.mask = mask.view_as(self.blocks)
        self.n_tokens = int(self.mask.sum())
        self.base_n_tokens = int(getattr(ds, "n_tokens", self.blocks.numel()))
        self.n_answers = n_answers

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, i: int) -> dict:
        ids = self.blocks[i]; labels = ids.clone(); labels[~self.mask[i]] = -100
        return {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}

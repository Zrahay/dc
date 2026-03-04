"""Benchmark DNABert on the GUE 'prom_core_all' task.

This script:

- Loads the GUE dataset split 'prom_core_all' from HuggingFace Datasets
- Extracts DNA sequences and integer labels
- Wraps them into DeepChem Dataset objects
- Trains DeepChem's DNABert model
- Evaluates F1 score and Matthews Correlation Coefficient (MCC) on the test split

Run as:

    python benchmark_dnabert_gue.py
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

import deepchem as dc
from deepchem.models.torch_models.dna_bert_model import DNABert

try:
    from datasets import load_dataset
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The 'datasets' library is required for this benchmark. "
        "Install it with `pip install datasets`."
    ) from e


def load_gue_prom_core_all() -> Tuple[dc.data.Dataset, dc.data.Dataset]:
    """Load GUE 'prom_core_all' split and return DeepChem train/test datasets.

    The underlying HF dataset is expected to have fields:
        - 'sequence': DNA sequence string
        - 'label': integer class
    """
    ds = load_dataset("leannmlindsey/GUE", "prom_core_all")

    if "train" not in ds:
        raise ValueError("Expected a 'train' split in GUE/prom_core_all.")
    if "test" not in ds:
        # Fall back to 'validation' if 'test' is missing
        if "validation" in ds:
            ds["test"] = ds["validation"]
        else:
            raise ValueError(
                "Expected a 'test' or 'validation' split in GUE/prom_core_all."
            )

    def hf_split_to_dc(split) -> dc.data.NumpyDataset:
        sequences = [ex["sequence"] for ex in split]
        labels = np.asarray([ex["label"] for ex in split], dtype=np.int64)

        # DNABert expects classification labels; DeepChem convention is (n_samples, n_tasks).
        y = labels.reshape(-1, 1)

        # X holds raw DNA strings; DNABert tokenizes them on-the-fly using its internal
        # HuggingFace DNABERT tokenizer, consistent with the TorchModel/HuggingFaceModel
        # integration pattern used in ChemBERTa.
        X = np.asarray(sequences, dtype=object)

        # Uniform weights and simple string ids
        w = np.ones_like(y, dtype=np.float32)
        ids = np.asarray([str(i) for i in range(len(sequences))], dtype=object)

        return dc.data.NumpyDataset(X=X, y=y, w=w, ids=ids)

    train_dc = hf_split_to_dc(ds["train"])
    test_dc = hf_split_to_dc(ds["test"])
    return train_dc, test_dc


def build_dnabert_model(model_dir: str) -> DNABert:
    """Construct a DNABert model configured for binary classification."""
    # DNABert internally uses the DNABERT-2 tokenizer via HuggingFace's
    # `AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")`.
    return DNABert(
        task="classification",
        tokenizer_path="zhihan1996/DNABERT-2-117M",
        n_tasks=1,
        model_dir=model_dir,
        batch_size=16,
    )


def main() -> None:
    dataset_name = "prom_core_all"

    print(f"Loading GUE dataset: {dataset_name} ...", flush=True)
    train_dataset, test_dataset = load_gue_prom_core_all()

    model_dir = os.path.join("dnabert_gue_runs", dataset_name)
    os.makedirs(model_dir, exist_ok=True)

    print("Building DNABert model ...", flush=True)
    model = build_dnabert_model(model_dir)

    # Define DeepChem metrics
    metric_f1 = dc.metrics.Metric(dc.metrics.f1_score, mode="classification")
    metric_mcc = dc.metrics.Metric(dc.metrics.matthews_corrcoef, mode="classification")

    print("Training DNABert ...", flush=True)
    # For a quick benchmark, use a small number of epochs. Adjust as needed.
    model.fit(train_dataset, nb_epoch=3)

    print("Evaluating on test split ...", flush=True)
    scores = model.evaluate(test_dataset, metrics=[metric_f1, metric_mcc])

    test_f1 = scores.get("f1_score", None)
    test_mcc = scores.get("matthews_corrcoef", None)

    print(f"Dataset: {dataset_name}")
    print(f"Test F1: {test_f1}")
    print(f"Test MCC: {test_mcc}")


if __name__ == "__main__":  # pragma: no cover
    main()


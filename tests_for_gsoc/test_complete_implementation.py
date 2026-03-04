# test_dna_bert_model.py

import os
import tempfile

import numpy as np
import pandas as pd
import deepchem as dc
import pytest

from deepchem.models.torch_models.dna_bert_model import DNABert


@pytest.mark.parametrize("task", ["regression"])
def test_dnabert_regression_tiny_dataset(task):
    """Smoke test that DNABert trains and predicts on a tiny DNA dataset.

    This specifically checks:
    - Integration with DeepChem Dataset (via CSVLoader + DummyFeaturizer)
    - That fit() runs without crashing
    - That predict() returns outputs of the right shape
    """

    # 1. Create a tiny in-memory DNA dataset
    sequences = [
        "ACGTACGTACGT",
        "GGGTTTAAACCC",
        "TATATATATATA",
        "CCCCGGGGAAAA",
    ]
    labels = [0.1, 0.9, -0.2, 0.5]

    df = pd.DataFrame({"sequence": sequences, "label": labels})

    with dc.utils.UniversalNamedTemporaryFile(mode="w") as tmpfile:
        df.to_csv(tmpfile.name, index=False)

        # 2. Use CSVLoader + DummyFeaturizer so Dataset.X holds raw strings
        loader = dc.data.CSVLoader(
            tasks=["label"],
            feature_field="sequence",
            featurizer=dc.feat.DummyFeaturizer(),
        )
        dataset = loader.create_dataset(tmpfile.name)

    # 3. Construct DNABert model
    # Use a small model_dir in a temp directory to avoid cluttering repo
    with tempfile.TemporaryDirectory() as model_dir:
        model = DNABert(
            task=task,
            tokenizer_path="zhihan1996/DNABERT-2-117M",
            n_tasks=1,
            model_dir=model_dir,
            batch_size=2,          # small batch for fast test
        )

        # 4. Run a very short training to ensure the pipeline works
        loss = model.fit(dataset, nb_epoch=1)
        # Just check loss is a finite float (no NaNs, no crashes)
        assert np.isfinite(loss)

        # 5. Run prediction and check shapes
        preds = model.predict(dataset)  # type: ignore[assignment]
        # preds should be (n_samples, n_tasks)
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == len(sequences)
        assert preds.shape[1] == 1


@pytest.mark.parametrize("task", ["classification"])
def test_dnabert_classification_tiny_dataset(task):
    """Smoke test that DNABert runs on a tiny DNA classification dataset.

    Checks that:
    - Model can be trained without crashing
    - Prediction logits have expected shape (n_samples, 2) for binary classification
    """

    sequences = [
        "ACGTACGTACGT",
        "GGGTTTAAACCC",
        "TATATATATATA",
        "CCCCGGGGAAAA",
    ]
    # Random binary labels; DeepChem convention is (n_samples, n_tasks)
    labels = np.random.choice([0, 1], size=(len(sequences), 1)).astype(np.int64)

    X = np.asarray(sequences, dtype=object)
    w = np.ones_like(labels, dtype=np.float32)
    ids = np.asarray([str(i) for i in range(len(sequences))], dtype=object)
    dataset = dc.data.NumpyDataset(X=X, y=labels, w=w, ids=ids)

    with tempfile.TemporaryDirectory() as model_dir:
        model = DNABert(
            task=task,
            tokenizer_path="zhihan1996/DNABERT-2-117M",
            n_tasks=1,
            model_dir=model_dir,
            batch_size=2,
        )

        loss = model.fit(dataset, nb_epoch=1)
        assert np.isfinite(loss)

        preds = model.predict(dataset)  # type: ignore[assignment]
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == len(sequences)
        # Binary classification head should output 2 logits by default
        assert preds.shape[1] == 2


def test_dnabert_mlm_tiny_dataset():
    """Smoke test for DNABert in masked language modeling (MLM) mode."""

    sequences = [
        "ACGTACGTACGT",
        "GGGTTTAAACCC",
        "TATATATATATA",
        "CCCCGGGGAAAA",
    ]

    # For MLM, labels are generated internally; we can set y=None.
    X = np.asarray(sequences, dtype=object)
    dataset = dc.data.NumpyDataset(X=X, y=None)

    with tempfile.TemporaryDirectory() as model_dir:
        model = DNABert(
            task="mlm",
            tokenizer_path="zhihan1996/DNABERT-2-117M",
            n_tasks=1,
            model_dir=model_dir,
            batch_size=2,
        )

        loss = model.fit(dataset, nb_epoch=1)
        assert np.isfinite(loss)


def test_dnabert_mtr_tiny_dataset():
    """Smoke test for DNABert in multitask regression (MTR) mode."""

    sequences = [
        "ACGTACGTACGT",
        "GGGTTTAAACCC",
        "TATATATATATA",
        "CCCCGGGGAAAA",
    ]
    # Two regression tasks
    labels = np.random.randn(len(sequences), 2).astype(np.float32)

    X = np.asarray(sequences, dtype=object)
    w = np.ones_like(labels, dtype=np.float32)
    ids = np.asarray([str(i) for i in range(len(sequences))], dtype=object)
    dataset = dc.data.NumpyDataset(X=X, y=labels, w=w, ids=ids)

    with tempfile.TemporaryDirectory() as model_dir:
        model = DNABert(
            task="mtr",
            tokenizer_path="zhihan1996/DNABERT-2-117M",
            n_tasks=2,
            model_dir=model_dir,
            batch_size=2,
        )

        loss = model.fit(dataset, nb_epoch=1)
        assert np.isfinite(loss)
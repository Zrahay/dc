import os
import pytest
import deepchem as dc
import pandas as pd


@pytest.fixture
def grover_batched_graph():
    from deepchem.utils.grover import BatchGroverGraph
    smiles = ['CC', 'CCC', 'CC(=O)C']

    fg = dc.feat.CircularFingerprint()
    featurizer = dc.feat.GroverFeaturizer(features_generator=fg)

    graphs = featurizer.featurize(smiles)
    batched_graph = BatchGroverGraph(graphs)
    return batched_graph


@pytest.fixture
def smiles_regression_dataset(tmpdir):
    smiles = [
        "CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F",
        "CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1"
    ]
    labels = [3.112, 2.432]
    df = pd.DataFrame(list(zip(smiles, labels)), columns=["smiles", "task1"])
    filepath = os.path.join(tmpdir, 'smiles.csv')
    df.to_csv(filepath)

    loader = dc.data.CSVLoader(["task1"],
                               feature_field="smiles",
                               featurizer=dc.feat.DummyFeaturizer())
    dataset = loader.create_dataset(filepath)
    return dataset


@pytest.fixture
def smiles_multitask_regression_dataset():
    cwd = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(cwd,
                              '../../tests/assets/multitask_regression.csv')

    loader = dc.data.CSVLoader(tasks=['task0', 'task1'],
                               feature_field='smiles',
                               featurizer=dc.feat.DummyFeaturizer())
    dataset = loader.create_dataset(input_file)
    return dataset


@pytest.fixture
def dna_regression_dataset(tmpdir):
    sequences = [
        "ACGTACGTACGT",
        "GGGTTTAAACCC",
        "TATATATATATA",
        "CCCCGGGGAAAA",
    ]
    labels = [0.1, 0.9, -0.2, 0.5]
    df = pd.DataFrame(list(zip(sequences, labels)),
                      columns=["sequence", "task1"])
    filepath = os.path.join(tmpdir, 'dna.csv')
    df.to_csv(filepath)

    loader = dc.data.CSVLoader(["task1"],
                                feature_field="sequence",
                                featurizer=dc.feat.DummyFeaturizer())
    dataset = loader.create_dataset(filepath)
    return dataset


@pytest.fixture
def dna_multitask_regression_dataset():
    import numpy as np
    sequences = [
        "ACGTACGTACGT",
        "GGGTTTAAACCC",
        "TATATATATATA",
        "CCCCGGGGAAAA",
    ]
    labels = np.array([[0.1, 0.3], [0.9, 0.2], [-0.2, 0.7],
                       [0.5, -0.1]]).astype(np.float32)
    X = np.asarray(sequences, dtype=object)
    w = np.ones_like(labels, dtype=np.float32)
    ids = np.asarray([str(i) for i in range(len(sequences))], dtype=object)
    dataset = dc.data.NumpyDataset(X=X, y=labels, w=w, ids=ids)
    return dataset


@pytest.fixture
def protein_classification_dataset(tmpdir):
    protein = [
        "M G L P V S W A P P A L W V L G C C A L L L S L W A",
        "M E V L E E P A P G P G G A D A A E R R G L R R L"
    ]
    labels = [0, 1]
    df = pd.DataFrame(list(zip(protein, labels)), columns=["protein", "task1"])
    filepath = os.path.join(tmpdir, 'protein.csv')
    df.to_csv(filepath)

    loader = dc.data.CSVLoader(["task1"],
                               feature_field="protein",
                               featurizer=dc.feat.DummyFeaturizer())
    dataset = loader.create_dataset(filepath)
    return dataset

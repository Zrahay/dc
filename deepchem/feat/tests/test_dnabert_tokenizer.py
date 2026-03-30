import pytest


@pytest.mark.torch
def test_dnabert_tokenizer_call():
    """Test __call__ method for the DNABertTokenizer, which is inherited from HuggingFace's PreTrainedTokenizerFast"""
    from deepchem.feat.dnabert_tokenizer import DNABertTokenizer
    sequences = ["ACGTACGTACGT", "GGGTTTAAACCC"]
    long_sequence = ["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"]
    featurizer = DNABertTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    embedding = featurizer(sequences, add_special_tokens=True, truncation=True)
    embedding_long = featurizer(long_sequence * 2,
                                add_special_tokens=True,
                                truncation=True)
    for emb in [embedding, embedding_long]:
        assert 'input_ids' in emb.keys() and 'attention_mask' in emb.keys()
        assert len(emb['input_ids']) == 2 and len(emb['attention_mask']) == 2


@pytest.mark.torch
def test_dnabert_tokenizer_featurize():
    """Test the .featurize method, which will convert the dictionary output to an array

    Checks that all DNA sequences are featurized and that each featurization
    contains input_ids and attention_mask
    """
    from deepchem.feat.dnabert_tokenizer import DNABertTokenizer
    sequences = ["ACGTACGTACGT", "GGGTTTAAACCC"]
    long_sequence = ["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"]
    featurizer = DNABertTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    max_length = 32
    feat_kwargs = {
        'add_special_tokens': True,
        'truncation': True,
        'padding': 'max_length',
        'max_length': max_length
    }
    feats = featurizer.featurize(sequences, **feat_kwargs)
    assert len(feats) == 2
    assert all([len(f) == 2 for f in feats])
    assert all([len(f[0]) == max_length for f in feats])

    long_feat = featurizer.featurize(long_sequence, **feat_kwargs)
    assert len(long_feat) == 1
    assert len(long_feat[0]) == 2  # the tokens and attention mask
    assert len(long_feat[0][0]) == max_length


@pytest.mark.torch
def test_dnabert_tokenizer_deterministic():
    """Test that tokenizing the same sequence twice produces identical results"""
    from deepchem.feat.dnabert_tokenizer import DNABertTokenizer
    sequence = "ACGTACGTACGT"
    featurizer = DNABertTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    output1 = featurizer(sequence)
    output2 = featurizer(sequence)

    assert output1['input_ids'] == output2['input_ids']
    assert output1['attention_mask'] == output2['attention_mask']
    assert len(output1['input_ids']) == len(output1['attention_mask'])
    assert len(output1['input_ids']) > 0

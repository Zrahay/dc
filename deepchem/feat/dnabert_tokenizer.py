from deepchem.feat import Featurizer
from typing import Dict, List
try:
    from transformers import PreTrainedTokenizerFast
except ModuleNotFoundError:
    raise ImportError(
        'Transformers must be installed for DNATokenizer to be used.'
    )
    pass


class DNABertTokenizer(PreTrainedTokenizerFast, Featurizer):
    """DNABERT-style tokenizer featurizer.

    This class wraps a HuggingFace fast tokenizer for DNABERT-style DNA
    sequence tokenization, while also conforming to DeepChem's
    :class:`Featurizer` interface. It can therefore be used in two ways:

    - As a standard HuggingFace tokenizer (via ``from_pretrained``, ``__call__``,
      ``encode``, ``decode``, etc.).
    - As a DeepChem featurizer that can be passed to loaders such as
      :class:`deepchem.data.CSVLoader` to convert raw DNA strings into
      tokenized features stored in :class:`deepchem.data.Dataset` objects.

    When used as a featurizer, a single DNA sequence string is mapped to
    a list of arrays (for example ``[input_ids, attention_mask]``) based
    on the underlying tokenizer configuration.

    Examples
    --------
    Create a DNABERT tokenizer featurizer from a pretrained checkpoint
    and featurize a list of DNA sequences.

    >>> from deepchem.feat import DNABertTokenizer
    >>> sequences = ["ACGTACGT", "GGGTTTAAA"]
    >>> featurizer = DNABertTokenizer.from_pretrained(
    ...     "zhihan1996/DNABERT-2-117M"
    ... )
    >>> encoding = featurizer(sequences,
    ...                       add_special_tokens=True,
    ...                       truncation=True,
    ...                       padding=True)

    Use as a DeepChem featurizer in a CSV loader.

    >>> import deepchem as dc
    >>> featurizer = DNABertTokenizer.from_pretrained(
    ...     "zhihan1996/DNABERT-2-117M"
    ... )
    >>> loader = dc.data.CSVLoader(tasks=["label"],
    ...                            feature_field="sequence",
    ...                            featurizer=featurizer)
    >>> dataset = loader.create_dataset("promoter_sequences.csv")

    Note
    ----
    This class requires the ``transformers`` library to be installed.
    """

    def __init__(self, **kwargs):
        """Initialize a DNABertTokenizer.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to
            :class:`transformers.PreTrainedTokenizerFast`. These are
            typically populated automatically when calling
            :meth:`from_pretrained` with a DNABERT-style tokenizer
            checkpoint.
        """
        super().__init__(**kwargs)
        return

    def _featurize(self, datapoint: str, **kwargs) -> List[List[int]]:
        """Featurize a single DNA sequence using the fast tokenizer.

        Parameters
        ----------
        datapoint : str
            DNA sequence string to tokenize.
        **kwargs
            Additional keyword arguments forwarded to the underlying
            tokenizer ``__call__`` method (for example ``padding``,
            ``truncation``, or ``max_length``).

        Returns
        -------
        List[List[int]]
            A list of tokenization outputs for the given datapoint.
            The exact contents depend on the tokenizer configuration,
            but commonly this will be a list containing arrays such as
            ``input_ids`` and ``attention_mask``.
        """
        encoding = list(self(datapoint, **kwargs).values())
        return encoding

    def __call__(self, *args, **kwargs) -> Dict[str, List[int]]:
        """Tokenize one or more DNA sequences.

        This forwards directly to
        :meth:`transformers.PreTrainedTokenizerFast.__call__` and
        preserves all standard HuggingFace tokenizer semantics.

        Returns
        -------
        Dict[str, List[int]] or BatchEncoding
            The standard HuggingFace tokenization output, typically
            containing fields such as ``input_ids`` and
            ``attention_mask``.
        """
        return super().__call__(*args, **kwargs)











from typing import Dict, Any, Tuple

from deepchem.models.torch_models.hf_models import HuggingFaceModel
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
)
from transformers import AutoTokenizer

try:
    import torch
    has_torch = True
except Exception:  # pragma: no cover
    has_torch = False


class DNABert(HuggingFaceModel):
    """DNABERT-style transformer model for DNA sequences.

    DNABert is a BERT architecture adapted for DNA sequence modeling. This
    class wraps a HuggingFace DNABERT-style model inside DeepChem's
    :class:`HuggingFaceModel`, providing a unified interface for pretraining
    and finetuning using the standard :class:`TorchModel` API.

    The model supports multiple tasks:

    - ``"mlm"``: masked language modeling on DNA sequences.
    - ``"mtr"``: multitask regression head on top of the BERT encoder.
    - ``"regression"``: standard regression head for property prediction.
    - ``"classification"``: single- or multi-task classification.

    Tokenization is handled by a DNABERT-compatible tokenizer loaded from the
    specified ``tokenizer_path``. Raw DNA strings are stored in
    :class:`deepchem.data.Dataset` objects (typically using a
    :class:`deepchem.feat.DummyFeaturizer`), and tokenization is performed
    on-the-fly in :meth:`_prepare_batch`.

    Parameters
    ----------
    task : str
        The learning task to use for DNABert. Supported values are:

        - ``"mlm"``: masked language modeling pretraining.
        - ``"mtr"``: multitask regression pretraining/finetuning.
        - ``"regression"``: regression finetuning.
        - ``"classification"``: classification finetuning.
    tokenizer_path : str, default "zhihan1996/DNABERT-2-117M"
        HuggingFace model or tokenizer identifier used to construct the
        tokenizer. This can be a model hub id (for example
        ``"zhihan1996/DNABERT-2-117M"``) or a local directory containing a
        saved tokenizer.
    n_tasks : int, default 1
        Number of prediction targets for multitask learning. This controls
        the number of output units in the regression or classification head.
    config : dict, default {}
        Optional configuration overrides for :class:`transformers.BertConfig`.
        These are merged with the default DNABERT configuration and can be
        used to customize architectural parameters such as hidden size or
        number of layers.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`deepchem.models.torch_models.HuggingFaceModel`, such as
        ``model_dir``, ``batch_size``, or logging options.

    Examples
    --------
    Pretrain DNABert on a masked language modeling task:

    >>> import deepchem as dc
    >>> from deepchem.models.torch_models.dna_bert import DNABert
    >>> # dataset.X should contain raw DNA strings
    >>> model = DNABert(task="mlm",
    ...                 tokenizer_path="zhihan1996/DNABERT-2-117M",
    ...                 model_dir="dnabert-mlm")
    >>> loss = model.fit(dataset, nb_epoch=1)

    Finetune DNABert for binary classification:

    >>> model = DNABert(task="classification",
    ...                 tokenizer_path="zhihan1996/DNABERT-2-117M",
    ...                 n_tasks=1,
    ...                 model_dir="dnabert-cls")
    >>> loss = model.fit(dataset, nb_epoch=1)
    >>> preds = model.predict(dataset)
    """

    def __init__(
        self,
        task: str,
        tokenizer_path: str = "zhihan1996/DNABERT-2-117M",
        n_tasks: int = 1,
        config: Dict[Any, Any] = {},
        **kwargs,
    ):
        self.n_tasks = n_tasks
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        dna_bert_config = BertConfig(vocab_size=tokenizer.vocab_size, **config)

        if task == "mlm":
            model = BertForMaskedLM(dna_bert_config)
        elif task == "mtr":
            dna_bert_config.problem_type = "regression"
            dna_bert_config.num_labels = n_tasks
            model = BertForSequenceClassification(dna_bert_config)
        elif task == "regression":
            dna_bert_config.problem_type = "regression"
            dna_bert_config.num_labels = n_tasks
            model = BertForSequenceClassification(dna_bert_config)
        elif task == "classification":
            if n_tasks == 1:
                dna_bert_config.problem_type = "single_label_classification"
            else:
                dna_bert_config.problem_type = "multi_label_classification"
                dna_bert_config.n_labels = n_tasks
            model = BertForSequenceClassification(dna_bert_config)
        else:
            raise ValueError("Invalid task specification for DNABert.")

        super(DNABert, self).__init__(
            model=model,
            task=task,
            tokenizer=tokenizer,
            **kwargs,
        )

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        """Prepare a batch of DNA sequences for DNABert.

        This method overrides :meth:`HuggingFaceModel._prepare_batch` to
        handle the different label types required by the various DNABert
        tasks. Raw DNA sequences are tokenized on-the-fly using the
        underlying HuggingFace tokenizer, and labels are cast to the
        appropriate PyTorch dtypes for regression or classification.

        Parameters
        ----------
        batch : tuple
            A tuple ``(inputs, labels, weights)`` produced by a
            :class:`deepchem.data.Dataset` iterator, where:

            - ``inputs`` is a list containing an array of DNA sequence
              strings.
            - ``labels`` is a list containing the label array or ``None``
              during prediction.
            - ``weights`` is a list containing the per-sample weights.

        Returns
        -------
        inputs : dict
            Dictionary of tokenized inputs suitable for passing to the
            underlying HuggingFace BERT model. For example, in MLM mode
            this includes masked ``input_ids`` and ``labels``; for
            supervised tasks it contains tokenized inputs and a ``labels``
            tensor of the correct dtype.
        labels : torch.Tensor or None
            Label tensor on the correct device for supervised tasks, or
            ``None`` for masked language modeling where the labels are
            embedded in the inputs.
        weights : list
            Sample weights passed through unchanged from the original
            batch.
        """
        genome_batch, y, w = batch
        tokens = self.tokenizer(
            genome_batch[0].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if self.task == "mlm":
            inputs_ids, labels = self.data_collator.torch_mask_tokens(
                tokens["input_ids"]
            )

            inputs = {
                "input_ids": inputs_ids.to(self.device),
                "labels": labels.to(self.device),
                "attention_mask": tokens["attention_mask"].to(self.device),
            }

            return inputs, None, w

        elif self.task in ["regression", "classification", "mtr"]:
            if y is not None:
                y = torch.from_numpy(y[0])
                if self.task in ["regression", "mtr"]:
                    y = y.float().to(self.device)
                elif self.task == "classification":
                    if self.n_tasks == 1:
                        y = y.long().to(self.device)
                    else:
                        y = y.float().to(self.device)
            for key, value in tokens.items():
                tokens[key] = value.to(self.device)

            inputs = {**tokens, "labels": y}
            return inputs, y, w
from typing import Dict, Any, Tuple
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification
)
from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
try:
    import torch
    has_torch = True
except:
    has_torch = False


class DNABert(HuggingFaceModel):
    """
    Docstrings for this class
    """
    def __init__(self,
                 task:str,
                 tokenizer_path,
                 n_tasks: int = 1,
                 config: Dict[Any, Any] = {},
                 **kwargs):
        self.n_tasks = n_tasks
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
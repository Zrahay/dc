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
                 tokenizer_path: str = "zhihan1996/DNABERT-2-117M",
                 n_tasks: int = 1,
                 config: Dict[Any, Any] = {},
                 **kwargs):
        self.n_tasks = n_tasks
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        dna_bert_config = BertConfig(vocab_size = tokenizer.vocab_size,
                                     **config)
        

        if task == "mlm":
            model = BertForMaskedLM(dna_bert_config)
        elif task == "mtr":
            dna_bert_config.problem_type = 'regression'
            dna_bert_config.num_labels = n_tasks
            model = BertForSequenceClassification(dna_bert_config)
        elif task == 'classification':
            if n_tasks == 1:
                dna_bert_config.problem_type = 'single_label_classification'
            else:
                dna_bert_config.problem_type = 'multi_label_classification'
                dna_bert_config.n_labels = n_tasks
            model = BertForSequenceClassification(dna_bert_config)
        else:
            raise ValueError('Invalid Task Specification')
        
        super(DNABert, self).__init__(model = model,
                               task = task,
                               tokenizer = tokenizer,
                               **kwargs)
        
        


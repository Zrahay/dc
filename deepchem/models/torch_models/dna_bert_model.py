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
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) # This line might cause ceratin issue so carefully look debug this line if some issues come up later on while writing some code
        dna_bert_config = BertConfig(vocab_size = tokenizer.vocab_size,
                                     **config)
        

        if task == "mlm":
            model = BertForMaskedLM(dna_bert_config)
        elif task == "mtr":
            dna_bert_config.problem_type = 'regression'
            dna_bert_config.num_labels = n_tasks
            model = BertForSequenceClassification(dna_bert_config)
        elif task == 'regression':
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
    


    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        """
        """
        genome_batch, y, w = batch
        tokens = self.tokenizer(genome_batch[0].tolist(),
                                padding = True,
                                truncation = True,
                                return_tensors = "pt")
        if self.task == "mlm":
            inputs, labels = self.data_collator.torch_mask_tokens(
                tokens['input_ids']
            )

            inputs = {
                'input_ids': inputs.to(self.device),
                'labels': labels.to(self.device),
                'attenion_mask': tokens['attention_mask'].to(self.device)
            }

            return inputs, None, w
        
        elif self.task in ['regression', 'classification', 'mtr']:
            if y is not None:
                y = torch.from_numpy(y[0])
                if self.task == 'regression' or self.task == 'mtr':
                    y = y.float().to(self.device)
                elif self.task == 'classification':
                    if self.n_tasks == 1:
                        y = y.long().to(self.device)
                    else:
                        y = y.float().to(self.device)
            for key, value in tokens.items():
                tokens[key] = value.to(self.device)

            inputs = {**tokens, 'labels':y}
            return inputs, y, w



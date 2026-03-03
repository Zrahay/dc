from deepchem.feat import Featurizer
from typing import Dict, List
try:
    from transformers import PreTrainedTokenizerFast
except ModuleNotFoundError:
    raise ImportError(
        'Transformers must be installed for DNATokenizer to be used.'
    )
    pass

# We need to add a custom function to make the pre processed DNA Sequence

class DNABertTokenizer(PreTrainedTokenizerFast, Featurizer):
    # Add Docstrings

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return
    

    def _featurize(self, datapoint: str, **kwargs) -> List[List[int]]:
        """Calculate encoding using HuggingFace's PreTrainedTokenizerFast"""
        encoding = list(self(datapoint, **kwargs).values())
        return encoding

    
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)











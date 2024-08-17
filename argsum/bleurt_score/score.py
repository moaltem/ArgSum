import torch
from .configuration_bleurt import BleurtConfig  # noqa: F401
from .modeling_bleurt import BleurtForSequenceClassification  # noqa: F401
from .tokenization_bleurt import BleurtTokenizer  # noqa: F401

def score(refs, cands):
    config = BleurtConfig.from_pretrained('models/metrics/BLEURT-20')
    model = BleurtForSequenceClassification.from_pretrained('models/metrics/BLEURT-20').to('mps')
    tokenizer = BleurtTokenizer.from_pretrained('models/metrics/BLEURT-20')

    model.eval()
    #score_list = []
    #for i in range(0, len(cands), batch_size):
    #    cands_list = cands[i: i + batch_size]
    #   refs_list = refs[i: i + batch_size]
    
    with torch.no_grad():
        inputs = tokenizer(refs, cands, max_length = 256, padding = 'longest', return_tensors = 'pt', truncation = True)
        inputs.to('mps')
        res = model(**inputs).logits.flatten().tolist()
        #score_list += res
    
    return res
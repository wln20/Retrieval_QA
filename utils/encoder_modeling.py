from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from abc import abstractmethod, ABC
import torch
import torch.nn.functional as F

class BaseEncoder(ABC):
    def __init__(self, model) -> None:
        pass
    
    @abstractmethod
    def encode(self, raw_txt):
        pass
        
class BertEncoder(BaseEncoder):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.model = AutoModel.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
    
    def encode(self, raw_txt):
        with torch.no_grad():
            docs_tokens = self.tokenizer(raw_txt, return_tensors='pt', padding=True).to(self.model.device)
            encoded_docs = torch.mean(self.model(**docs_tokens).last_hidden_state, dim=1).cpu().numpy().astype('float32')
        self.num_items = encoded_docs.shape[0]
        self.embed_dim = encoded_docs.shape[1]
        return (encoded_docs, self.num_items, self.embed_dim)

class BaichuanEncoder(BaseEncoder):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    
    def encode(self, raw_txt):
        with torch.no_grad():
            docs_tokens = self.tokenizer(raw_txt, return_tensors='pt', padding=True).to(self.model.device)
            encoded_docs = torch.mean(self.model(**docs_tokens, output_hidden_states=True).hidden_states[-1], dim=1).cpu().numpy().astype('float32')
        self.num_items = encoded_docs.shape[0]
        self.embed_dim = encoded_docs.shape[1]
        return (encoded_docs, self.num_items, self.embed_dim)     

class LlamaEncoder(BaseEncoder):
    def __init__(self, model) -> None:
        super().__init__(model)
    
    def encode(self, raw_txt):
        pass

class SBertEncoder(BaseEncoder):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
    
    def encode(self, raw_txt):
        # Adapted from https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        docs_tokens = self.tokenizer(raw_txt, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**docs_tokens)
        encoded_docs = mean_pooling(model_output, docs_tokens['attention_mask'])
        encoded_docs = F.normalize(encoded_docs, p=2, dim=1)

        self.num_items = encoded_docs.shape[0]
        self.embed_dim = encoded_docs.shape[1]    
        return (encoded_docs, self.num_items, self.embed_dim)         
    

encoder_list = {
    "BertForMaskedLM": BertEncoder,
    "BaichuanForCausalLM": BaichuanEncoder,
    "LlamaForCausalLM": LlamaEncoder,
    "SBert": SBertEncoder
}
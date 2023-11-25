from abc import abstractmethod, ABC
import torch

class BaseEncoder(ABC):
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
    
    @abstractmethod
    def encode(self, raw_txt):
        pass
        

class BertEncoder(BaseEncoder):
    def __init__(self, model, tokenizer) -> None:
        super().__init__(model, tokenizer)
    
    def encode(self, raw_txt):
        with torch.no_grad():
            docs_tokens = self.tokenizer(raw_txt, return_tensors='pt', padding=True).to(self.model.device)
            encoded_docs = torch.mean(self.model(**docs_tokens).last_hidden_state, dim=1).cpu().numpy().astype('float32')
        self.num_items = encoded_docs.shape[0]
        self.embed_dim = encoded_docs.shape[1]
        return (encoded_docs, self.num_items, self.embed_dim)

class BaichuanEncoder(BaseEncoder):
    def __init__(self, model, tokenizer) -> None:
        super().__init__(model, tokenizer)
    
    def encode(self, raw_txt):
        with torch.no_grad():
            docs_tokens = self.tokenizer(raw_txt, return_tensors='pt', padding=True).to(self.model.device)
            encoded_docs = encoded_docs = torch.mean(self.model(**docs_tokens, output_hidden_states=True).hidden_states[-1], dim=1).cpu().numpy().astype('float32')
        self.num_items = encoded_docs.shape[0]
        self.embed_dim = encoded_docs.shape[1]
        return (encoded_docs, self.num_items, self.embed_dim)      


class SentenceBertEncoder(BaseEncoder):
    def __init__(self, model, tokenizer) -> None:
        super().__init__(model, tokenizer)
    
    def encode(self, raw_txt):
        pass
            
    


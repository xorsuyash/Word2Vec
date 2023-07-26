import torch
import torch.nn as nn 

from constants import EMBED_DIMENSION,EMBED_MAX_NORM


class CBOW_Model(nn.Module):
    
    def __init__(self,vocab_size:int):
        super(CBOW_Model,self).__init__()
        
        self.embeddings=nn.Embedding(num_embeddings=vocab_size,
                                     embedding_dim=EMBED_DIMENSION,
                                     max_norm=EMBED_MAX_NORM,)
        
        self.linear= nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )
        
    def forward(self,x):
        
        x=self.embeddings(x)
        x=x.mean(axis=1)
        
        x=self.linear(x)
        
        return x 
    
class SkipGram_Model(nn.Module):
  
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x
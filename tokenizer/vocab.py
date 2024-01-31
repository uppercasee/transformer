from collections import defaultdict, Counter
class Vocab:    
    def __init__(self, tokens=None):        
        self.idx_to_token = list()        
        self.token_to_idx = dict()         
        if tokens is not None:            
            if "<unk>" not in tokens:                
                tokens = tokens + ["<unk>"]            
            for token in tokens: 
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx['<unk>']
        
    @classmethod    
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens  \
                          else [])
        uniq_tokens += [token for token, freq in token_freqs.items() \
                          if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)
            
    def __len__(self):        
        return len(self.idx_to_token)
    def __getitem__(self, token):        
        return self.token_to_idx.get(token, self.unk)
    def convert_tokens_to_ids(self, tokens):        
        return [self[token] for token in tokens]
    def convert_ids_to_tokens(self, indices):        
        return [self.idx_to_token[index] for index in indices]
    
    def save_vocab(vocab, path):    
        with open(path, 'w') as writer:
            writer.write("\n".join(vocab.idx_to_token))
    def read_vocab(path):
        with open(path, 'r') as f:
            tokens = f.read().split('\n')
        return Vocab(tokens)


if __name__ == "__main__":
    # Creating a vocabulary from a list of sentences
    sentences = [["hello", "world"], ["example", "sentence"]]
    vocab = Vocab.build(sentences, min_freq=1)
    
    # Accessing vocabulary properties
    print(len(vocab))  # Number of tokens in the vocabulary
    print(vocab["hello"])  # Index of the token "hello"
    
    # Converting tokens to indices and vice versa
    print(vocab.convert_tokens_to_ids(["example", "world"]), vocab.convert_ids_to_tokens([1, 3]))

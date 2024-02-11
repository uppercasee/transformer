from collections import defaultdict, Counter


class Vocab:
    """
    A vocabulary object to convert tokens to indices and vice versa.

    Args:
        tokens: list of tokens to initialize the vocabulary. If None, the vocabulary will be empty.
    """

    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()
        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx["<unk>"]

    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        """
        Build a vocabulary from a list of sentences.

        Args:
            text: list of sentences, where each sentence is a list of tokens.
            min_freq: minimum frequency of a token for it to be included in the vocabulary.
            reserved_tokens: list of reserved tokens that will be added to the vocabulary.

        Returns:
            vocab: a vocabulary instance built from the provided text.
        """
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [
            token
            for token, freq in token_freqs.items()
            if freq >= min_freq and token != "<unk>"
        ]
        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        """
        Convert a list of tokens to a list of indices.

        Args:
            tokens: list of tokens to convert.

        Returns:
            indices: list of indices corresponding to the input tokens.
        """
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        """
        Convert a list of indices to a list of tokens.

        Args:
            indices: list of indices to convert.

        Returns:
            tokens: list of tokens corresponding to the input indices.
        """
        return [self.idx_to_token[index] for index in indices]

    def save_vocab(vocab, path):
        """
        Save a vocabulary to a file.

        Args:
            vocab: the vocabulary to save.
            path: the location to save the vocabulary to.

        Returns:
            None
        """
        with open(path, "w") as writer:
            writer.write("\n".join(vocab.idx_to_token))

    def read_vocab(path):
        """
        Read a vocabulary from a file.

        Args:
            path: the location to read the vocabulary from.

        Returns:
            vocab: a vocabulary instance read from the provided file.
        """
        with open(path, "r") as f:
            tokens = f.read().split("\n")
        return Vocab(tokens)


if __name__ == "__main__":
    # Creating a vocabulary from a list of sentences
    sentences = [["hello", "world"], ["example", "sentence"]]
    vocab = Vocab.build(sentences, min_freq=1)

    # Accessing vocabulary properties
    print(len(vocab))  # Number of tokens in the vocabulary
    print(vocab["hello"])  # Index of the token "hello"

    # Converting tokens to indices and vice versa
    print(
        vocab.convert_tokens_to_ids(["example", "world"]),
        vocab.convert_ids_to_tokens([1, 3]),
    )

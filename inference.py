from model import Encoder, Decoder, Seq2Seq
import torch
from torchtext.legacy.data import Field
import joblib


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_len = 150


class Inference():

    def __init__(self, device, max_len):
        self.model = None
        self.device = device
        self.DE = None
        self.EN = None
        self.max_len = max_len

    def load(self, path='save/seq2seq.pt'):

        # set parameters
        de_size, en_size = 8014, 10004
        embed_size = 256
        hidden_size = 512

        # initialize model
        encoder = Encoder(de_size, embed_size, hidden_size,
                          n_layers=2, dropout=0.5)
        decoder = Decoder(embed_size, hidden_size, en_size,
                          n_layers=1, dropout=0.5)
        self.model = Seq2Seq(encoder, decoder).to(self.device)

        # load model from checkpoint
        self.model.load_state_dict(torch.load(path)['model_state_dict'])
        self.model.eval()

        # load vocabulary

        self.DE = Field(tokenize='spacy',
                        tokenizer_language='de_core_news_sm',
                        include_lengths=True)
        self.DE.__setstate__(joblib.load('../data/DE.state'))

        self.EN = Field(tokenize='spacy',
                        tokenizer_language='en_core_web_sm',
                        include_lengths=True)
        self.EN.__setstate__(joblib.load('../data/EN.state'))

    def string2idx(self, text):
        '''convert text to list of indexes'''
        prep = self.DE.tokenize(text)
        post, _ = self.DE.process([prep])
        return post  # T*B(1)

    def infer(self, text):
        INPUT = self.string2idx(text).to(device)
        OUTPUT = self.model(INPUT, self.max_len)

        decoded_idx = []
        for t in OUTPUT:
            _, topi = t.topk(1)
            if topi.item() == 3:  # EN.vocab.stoi['<eos>']
                break
            decoded_idx.append(topi.item())

        OUTPUT = ' '.join([self.EN.vocab.itos[i] for i in decoded_idx[1:]])
        return OUTPUT


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len = 150
    infer = Inference(device=device, max_len=max_len)
    infer.load()
    infer.infer(
        'Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.')

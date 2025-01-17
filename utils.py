import re
import spacy
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import Multi30k

url = re.compile('(<url>.*</url>)')
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def tokenize_de(text):
    global spacy_de
    return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]


def tokenize_en(text):
    global spacy_en
    return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]


def load_dataset(batch_size):
    DE = Field(tokenize=tokenize_de, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN))
    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN

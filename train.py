import os
import math
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from tqdm import tqdm
# from utils import load_dataset
import pickle

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=3,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of batch size')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    '''
    g ← ∂C/∂W

    if ‖g‖ ≥ max_threshold or ‖g‖ ≤ min_threshold then

    g ← threshold (accordingly)'''
    return p.parse_args()


def evaluate(model, val_iter, vocab_size, \
            # DE, EN, \
            device):
    with torch.no_grad():
        model.eval()
        # pad = EN.vocab.stoi['<pad>'] # 1
        pad = 1
        total_loss = 0
        for b in range(len(val_iter[0])):
            # src, len_src = batch.src
            # trg, len_trg = batch.trg
            src, trg = val_iter[0][b], val_iter[1][b]
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)
            loss = F.nll_loss(output[1:].view(-1, vocab_size),
                                   trg[1:].contiguous().view(-1),
                                   ignore_index=pad)
            total_loss += loss.data.item()
        return total_loss / len(val_iter)


def train(e, model, optimizer, train_iter, vocab_size, grad_clip,\
            #  DE, EN,\
             device):
    model.train()
    total_loss = 0
    # pad = EN.vocab.stoi['<pad>']
    pad = 1
    for b in range(len(train_iter[0])):
        # src, len_src = batch.src
        # trg, len_trg = batch.trg
        src, trg = train_iter[0][b], train_iter[1][b]
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data.item()

        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            total_loss = 0

def lazy_load(data_name:str):
    with open('../data/sample/%sin.pkl' %data_name, 'rb') as handle:
        input = pickle.load(handle)
    with open('../data/sample/%sout.pkl' %data_name, 'rb') as handle:
        output = pickle.load(handle)    
    return (input, output)

def main():
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("[!] preparing dataset...")
    # train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size)
    train_iter, val_iter, test_iter = lazy_load('train'), lazy_load('val'), lazy_load('test')
    # de_size, en_size = len(DE.vocab), len(EN.vocab)
    de_size, en_size = 66, 64
    # print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
    #       % (len(train_iter), len(train_iter.dataset),
    #          len(test_iter), len(test_iter.dataset)))
    # print("[DE_vocab]:%d [en_vocab]:%d" % (de_size, en_size))

    print("[!] Instantiating models...")
    encoder = Encoder(de_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, en_size,
                      n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).to(device)
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    best_val_loss = None
    for e in tqdm(range(1, args.epochs+1)):
        train(e, seq2seq, optimizer, train_iter,
              en_size, args.grad_clip, device)
        val_loss = evaluate(seq2seq, val_iter, en_size, device)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
              % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), './.save/seq2seq_%d.pt' % (e))
            best_val_loss = val_loss
    test_loss = evaluate(seq2seq, test_iter, en_size, device)
    print("[TEST] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)

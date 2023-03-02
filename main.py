from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
from torchtext.utils import download_from_url, extract_archive
import io
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from layers.transformer import Transformer


MODEL_PATH = 'translator-transformer.pth'#'translator-transformer-bigger.pth' #'translator-transformer.pth'
PAD_IDX = 1
EOS_IDX = 3
BOS_IDX = 2
D_MODEL = 256
N_LAYERS = 6
CNT_FFN_UNITS = 1024
N_HEADS = 8
DROPOUT_RATE = 0.1

if torch.cuda.is_available():
    print('Will be using cuda.')
else:
    print('Will be using cpu.')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def loss_function(target, pred):
    mask = torch.not_equal(target, PAD_IDX)
    pred = pred.transpose(-2, -1)
    loss_ = nn.functional.cross_entropy(pred, target, reduce=False, reduction='none')

    mask = mask.to(dtype=loss_.dtype)
    loss_ *= mask

    return torch.mean(loss_)


def train(train_dataset, valid_dataset, transformer: Transformer, n_epochs: int, print_every: int = 50):
    writer = SummaryWriter()

    lr_mul = 0.0001

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr_mul, betas=(0.9, 0.98), eps=1e-9)
    best_valid_loss = 1e10

    for e in tqdm(range(n_epochs)):
        print(f'initiating epoch {e+1} out of {n_epochs}')
        losses = []
        transformer.train()
        for (batch, (enc_inputs, targets)) in enumerate(train_dataset):
            enc_inputs = enc_inputs.to(device)
            dec_inputs = targets[:, :-1].to(device)
            dec_outputs_real = targets[:, 1:].to(device)
            pred = transformer(enc_inputs, dec_inputs)

            #scheduler.zero_grad()
            optimizer.zero_grad()
            loss = loss_function(dec_outputs_real, pred)
            losses.append(loss.item())
            loss.backward()
            # scheduler.step_and_update_lr()
            optimizer.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('Training loss', avg_loss, e)

        transformer.eval()
        valid_losses = []
        for (batch, (enc_inputs, targets)) in enumerate(valid_dataset):
            enc_inputs = enc_inputs.to(device)
            dec_inputs = targets[:, :-1].to(device)
            dec_outputs_real = targets[:, 1:].to(device)
            pred = transformer(enc_inputs, dec_inputs)
            val_loss = loss_function(dec_outputs_real, pred)
            valid_losses.append(val_loss.item())

        avg_valid_loss = sum(valid_losses) / len(valid_losses)
        writer.add_scalar('Validation loss', avg_valid_loss, e)

        if (e+1) % print_every == 0:
            print('average train batch loss:', avg_loss)
            print('average validation batch loss:', avg_valid_loss)

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            print('saving model')
            torch.save(transformer.state_dict(), MODEL_PATH)

    writer.flush()


def test(dataset, transformer, de_vocab, en_vocab, en_tokenizer):
    for (batch, (enc_inputs, targets)) in enumerate(dataset):
        assert (enc_inputs < len(de_vocab)).all(), "target: {} invalid".format(enc_inputs)

        for i, t in enumerate(targets):
            t = ''
            for idx in targets[i]:
                t += (en_vocab.lookup_token(idx) + ' ')
            print('target', t)

            decoded = ''
            pred_idx = None
            while pred_idx != en_vocab['<eos>']:
                inputs_cur = [en_vocab[e] for e in en_tokenizer(decoded)]
                inputs_cur.insert(0, en_vocab['<bos>'])
                dec_inputs_cur = torch.tensor(inputs_cur, device=device)
                di = pad_sequence([dec_inputs_cur], True, padding_value=PAD_IDX)
                pred = transformer(enc_inputs[i][None, :].to(device), di)
                pred_last = pred[:, -1, :]
                pred_idx = pred_last.argmax().item()
                decoded += (en_vocab.lookup_token(pred_idx) + ' ')

            print('predicted', decoded)


def main(b_train=True):
    BATCH_SIZE = 128

    url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
    train_urls = ('train.de.gz', 'train.en.gz')
    val_urls = ('val.de.gz', 'val.en.gz')
    test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

    train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
    val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
    test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

    de_tokenizer = get_tokenizer('spacy', language='de')
    en_tokenizer = get_tokenizer('spacy', language='en')

    def build_vocab(filepath, tokenizer):
        counter = Counter()
        with io.open(filepath, encoding="utf8") as f:
            for string_ in f:
                counter.update(tokenizer(string_))
        return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
    de_vocab.set_default_index(de_vocab['<unk>'])
    en_vocab = build_vocab(train_filepaths[1], en_tokenizer)
    en_vocab.set_default_index(en_vocab['<unk>'])

    global PAD_IDX
    global BOS_IDX
    global EOS_IDX
    PAD_IDX = de_vocab['<pad>']
    BOS_IDX = de_vocab['<bos>']
    EOS_IDX = de_vocab['<eos>']

    def data_process(filepaths):
        raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
        raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
        data = []
        for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
            de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de)],
                                      dtype=torch.long)
            en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)],
                                      dtype=torch.long)
            data.append((de_tensor_, en_tensor_))
        return data

    train_data = data_process(train_filepaths)
    val_data = data_process(val_filepaths)
    test_data = data_process(test_filepaths)

    def generate_batch(data_batch):
        de_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:
            de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
            en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        de_batch = pad_sequence(de_batch, padding_value=PAD_IDX, batch_first=True)
        en_batch = pad_sequence(en_batch, padding_value=PAD_IDX, batch_first=True)
        return de_batch, en_batch

    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=generate_batch)
    valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=generate_batch)
    test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                           shuffle=True, collate_fn=generate_batch)

    num_words_inputs = len(de_vocab)
    num_words_outputs = len(en_vocab)

    transformer = Transformer(vocab_size_enc=num_words_inputs,
                              vocab_size_dec=num_words_outputs,
                              d_model=D_MODEL,
                              n_layers=N_LAYERS,
                              cnt_ffn_units=CNT_FFN_UNITS,
                              n_heads=N_HEADS,
                              dropout_rate=DROPOUT_RATE)
    transformer.to(device)
    if b_train:
        transformer.train(True)
        train(train_dataset=train_iter, valid_dataset=valid_iter, transformer=transformer, n_epochs=20, print_every=1)
    else:
        transformer.load_state_dict(torch.load(MODEL_PATH))
        transformer.eval()
        test(dataset=test_iter, transformer=transformer, de_vocab=de_vocab, en_vocab=en_vocab, en_tokenizer=en_tokenizer)


if __name__ == '__main__':
    main(b_train=True)

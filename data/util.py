import random, re, os
from data.prompt_dataset import * # commented when __name__ == "__main__"
from data.plot_dataset import * # commented when __name__ == "__main__"
import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from unidecode import unidecode
import functools
from rake_nltk import Rake
import urllib, sys
import urllib.request
import json, re
import numpy as np
from scipy.spatial.distance import cdist
from bert_serving.client import BertClient
from tqdm import trange


# for WI dataset
def paragraph_segmentation(data_dir='./data/', sep_factor=1.0):
    print('Loading wikiplot dataset...')
    data_plots = os.path.join(data_dir, 'wikiPlots/plots')
    with open(data_plots, errors='ignore') as fp:
        plots = fp.readlines()

    stories = []
    story = []
    for line in plots:
        if line == '<EOS>\n':
            stories.append(story[:100] if len(story) > 100 else story)
            story = []
        else:
            story.append(line.strip())

    with open(os.path.join(data_dir, 'wikiPlots/plots_paragraph'), 'w') as filehandle:
        for i in trange(len(stories)):
            story = stories[i]
            if len(story) == 0:
                filehandle.write('\n')
            else:
                # 4GPU, 15 hour: bert-serving-start -model_dir ./data/cased_L-12_H-768_A-12 -num_worker=16 -max_seq_len=NONE -port 7790 -port_out 7791
                bc = BertClient(check_length=False, port=7790, port_out=7791)
                article_vec = bc.encode(story)
                sent_sim_score = np.diag(cdist(article_vec[:-1], article_vec[1:], 'cosine'))
                mean_score, std_score = np.mean(sent_sim_score), np.std(sent_sim_score)

                para_list, cur_para = [], story[0]
                for i, score in enumerate(sent_sim_score):
                    if score < mean_score - std_score * sep_factor:
                        para_list.append(cur_para)
                        cur_para = story[i + 1]
                    else:
                        cur_para += ' ' + story[i + 1]
                para_list.append(cur_para)

                # avoid too short paragraph
                story_para = para_list[0]
                for p in para_list[1:]:
                    if len(p) < 114:
                        story_para += p
                    else:
                        story_para += '<newline><newline>' + p
                filehandle.write('%s\n' % story_para)


class Preprocessor():
    def __init__(self):
        self.fn = None

    def make_fn(self):
        raise NotImplementedError()

    def __call__(self, x):
        try:
            if self.fn is None:
                self.fn = self.make_fn()
            x = self.fn(x)
            return x
        except Exception as e:
            print('Error in preprocessing', repr(e))
            raise e


class Preprocessor_train(Preprocessor):
    def __init__(self, tokenizer, seq_len, model_type):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.model_type = model_type

    def make_fn(self):
        return compose(
            insert_keywords(self.tokenizer, self.model_type),
            lambda x: self.tokenizer.encode(x),
            prefix_truncate(self.seq_len)
        )


class Preprocessor_test(Preprocessor):
    def __init__(self, tokenizer, seq_len, model_type):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.model_type = model_type

    def make_fn(self):
        return compose(
            insert_keywords_test(self.tokenizer, self.model_type),
            lambda input: ([self.tokenizer.encode(x) for x in input[0]], [self.tokenizer.encode(y) for y in input[1]]),
        )


def compose(*functions):
    """ Executes a list of functions in order """
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)


def wp_preprocess(text):
    # Standardize some symbols
    text = text.replace('<newline>', '\n')
    text = text.replace('``', '"')
    text = text.replace("''", '"')
    # Detokenize
    text = re.sub(' +', ' ', text)  # replace multiple ' ' as one
    text = re.sub(' (\'|\.|\,|\:|\?|\!|;)', '\g<1>', text)  # remove ' ' before ,
    text = re.sub('" ([^"]*) "', '"\g<1>"', text)  # remove ' ' before and after ", " a " -> "a"
    text = text.replace(" n't", "n't")
    return text


def prefix_truncate(window):
    """ truncates text to the prefix window size """

    def f(text):
        if len(text) > window:
            text = text[:window]
        return text

    return f


def detect_dialog(t):
    if t.startswith('"') or t.startswith("'") or t.startswith("``") or t.startswith("`") or t.startswith(
            "''") or t.startswith("'") or t.startswith('“') or t.startswith('’') or t.startswith("‘") or t.startswith(
            '”'):
        return True
    else:
        return False


# # for WP dataset
def get_paragraph(story):
    # split as paragraphs
    # re.split("( <newline>){2,}", story) will keep ' <newline>' delimeter
    p = [x.strip() for x in re.split("( <newline>){2,}", story) if x != ' <newline>']

    # add dialog to preceding paragraph
    pp = [p[0]]
    for ii in range(1, len(p)):
        if detect_dialog(p[ii]) or len(p[ii]) < 114:
            pp[-1] = pp[-1] + ' <newline> ' + p[ii]
        else:
            pp.append(p[ii])
    pp = [wp_preprocess(pt) for pt in pp]

    return pp


def extract_keywords(text, r):
    r.extract_keywords_from_text(text)
    # 114 2, +1 per 228, add one key per 2 sentences, which is 114 in length
    num = min(5, max(2, int(len(text) / 228.0 + 1.5)))
    key = [re.sub(' (\'|\.|\,|\:|\?|\!|;)', '\g<1>', k.strip('\'.,:?!;" ')) for k in r.get_ranked_phrases()[:num]]
    return key


def insert_keywords(tokenizer, model_type):
    def f(text_raw_dict):
        # 'prompt' in text_raw_dict --> wp dataset; 'title' in text_raw_dict --> wi dataset
        summary = text_raw_dict['prompt'] if 'prompt' in text_raw_dict else text_raw_dict['title']
        story = text_raw_dict['story']

        if 'title' in text_raw_dict:
            pp = story.split('<newline><newline>')
        else:
            pp = get_paragraph(story)

        if model_type == 'm':
            # extract keywords
            r = Rake(min_length=1, max_length=4)
            keys = [extract_keywords(text, r) for text in pp]
            keys_str = [tokenizer.cls_token + tokenizer.sep_token.join(key) + tokenizer.mask_token for key in keys]
            keys_str[0] += tokenizer.eos_token
            story_inserted = ''.join([k + pt for k, pt in zip(keys_str, pp)])
            text = tokenizer.pad_token + summary + story_inserted + tokenizer.eos_token
        elif model_type == 'b0':
            # baseline 1, gpt2
            text = tokenizer.pad_token + summary + tokenizer.eos_token + '\n\n'.join(pp) + tokenizer.eos_token
        elif model_type == 'b1':
            # baseline 2, gpt2 ablation
            r = Rake(min_length=1, max_length=4)
            keys = [extract_keywords(text, r) for text in pp]
            keys_str = tokenizer.cls_token + tokenizer.sep_token.join([k for key in keys for k in key]) + tokenizer.mask_token
            text = tokenizer.pad_token + summary + keys_str + tokenizer.eos_token + '\n\n'.join(pp) + tokenizer.eos_token
        else:
            raise Exception('Model type not implemented.')

        return text

    return f


# a little bit different with insert_keywords, return keywords as list
def insert_keywords_test(tokenizer, model_type):
    def f(text_raw_dict):
        # 'prompt' in text_raw_dict --> wp dataset; 'title' in text_raw_dict --> wi dataset
        summary = text_raw_dict['prompt'] if 'prompt' in text_raw_dict else text_raw_dict['title']
        story = text_raw_dict['story']

        if 'title' in text_raw_dict:
            pp = story.split('<newline><newline>')
        else:
            pp = get_paragraph(story)

        if model_type == 'm':
            # extract keywords
            r = Rake(min_length=1, max_length=4)
            keys = [extract_keywords(text, r) for text in pp]
            # note that tokenizer.cls_token is added when decoding, not here
            keys_str = [tokenizer.sep_token.join(key) + tokenizer.mask_token for key in keys]
            keys_str[0] = tokenizer.pad_token + summary + tokenizer.cls_token + keys_str[0] + tokenizer.eos_token
            return (keys_str, pp)
        elif model_type == 'b0':
            # baseline 1, gpt2
            keys_str = []
            keys_str.append(tokenizer.pad_token + summary + tokenizer.eos_token)
            return (keys_str, pp)
        elif model_type == 'b1':
            # # baseline 2, gpt2 ablation
            r = Rake(min_length=1, max_length=4)
            keys = [extract_keywords(text, r) for text in pp]
            keys_str = []
            keys_str.append(tokenizer.pad_token + summary + tokenizer.cls_token + tokenizer.sep_token.join([k for key in keys for k in key]) + tokenizer.mask_token + tokenizer.eos_token)
            return (keys_str, pp)
        else:
            raise Exception('Model type not implemented.')

    return f


def collate_fn_masked(samples):
    """ Creates a batch out of samples """
    max_len = max(map(len, samples))
    # Zero pad mask
    x_mask = torch.ByteTensor([[1] * len(x) + [0] * (max_len - len(x)) for x in samples])
    # tokenizer.convert_tokens_to_ids('<|startoftext|>') = 50257, endoftext 50256, use 50257 here causes errors!!
    x = torch.LongTensor([x + [50256] * (max_len - len(x)) for x in samples])
    return x[:, :-1], x[:, 1:].contiguous(), x_mask[:, 1:]


def collate_fn_test(samples):
    """ Creates a batch out of samples """
    context = [k[0] for k, s in samples]
    keys = [k[1:] for k, s in samples]
    storys = [s for k, s in samples]

    max_len = max(map(len, context))
    # Zero pad mask
    context_mask = torch.ByteTensor([[0] * (max_len - len(x)) + [1] * len(x) for x in context])
    # tokenizer.convert_tokens_to_ids('<|startoftext|>') = 50257, endoftext 50256, use 50257 here causes errors!!
    context = torch.LongTensor([[50256] * (max_len - len(x)) + x for x in context])
    return context, context_mask[:, :-1], keys, storys


def prepare_dataset(data_dir, dataset_name, tokenizer, train_bsz, train_seq_len, val_bsz, val_seq_len, test_bsz=1,
                    test_seq_len=1024, model_type='m', num_workers=1, make_train=True, make_val=True, make_test=False):
    # data_dir, dataset_name, tokenizer, train_bsz, train_seq_len, val_bsz, val_seq_len, num_workers = args.data_dir, args.dataset, tokenizer, batch_schedule[cur_b_schedule][0], batch_schedule[cur_b_schedule][1], batch_schedule[-1][0], batch_schedule[-1][1], args.workers

    loaders = []
    if dataset_name == 'wp':
        train_collate_fn = collate_fn_masked
        val_collate_fn = collate_fn_masked
        test_collate_fn = collate_fn_test

        if make_train:
            train_preproc = Preprocessor_train(tokenizer, train_seq_len, model_type)
            d_train = PromptDataset(
                os.path.join(data_dir, 'writingPrompts/train.wp_source'),
                os.path.join(data_dir, 'writingPrompts/train.wp_target'),
                train_preproc)
            print('Train dataset size', len(d_train))
            loaders.append(data.DataLoader(d_train,
                                           # sampler=DistributedSampler(d_train) if distributed else None,
                                           batch_size=train_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=train_collate_fn) if d_train else None)
        if make_val:
            val_preproc = Preprocessor_train(tokenizer, val_seq_len, model_type)
            d_val = PromptDataset(
                os.path.join(data_dir, 'writingPrompts/valid.wp_source'),
                os.path.join(data_dir, 'writingPrompts/valid.wp_target'),
                val_preproc)
            print('Val dataset size', len(d_val))
            loaders.append(data.DataLoader(d_val,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=val_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=val_collate_fn) if d_val else None)
        if make_test:
            test_preproc = Preprocessor_test(tokenizer, test_seq_len, model_type)
            d_test = PromptDataset(
                os.path.join(data_dir, 'writingPrompts/test.wp_source'),
                os.path.join(data_dir, 'writingPrompts/test.wp_target'),
                test_preproc, sort=True)
            print('Test dataset size', len(d_test))
            loaders.append(data.DataLoader(d_test,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=test_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=test_collate_fn) if d_test else None)
    elif dataset_name == 'wi':
        train_collate_fn = collate_fn_masked
        val_collate_fn = collate_fn_masked
        test_collate_fn = collate_fn_test

        print('Loading wikiplot dataset...')
        data_plots = os.path.join(data_dir, 'wikiPlots/plots_paragraph')
        data_titles = os.path.join(data_dir, 'wikiPlots/titles')
        with open(data_plots, errors='ignore') as fp:
            plots = fp.readlines()
        with open(data_titles, errors='ignore') as ft:
            titles = ft.readlines()

        texts = [(t, p) for t, p in zip(titles, plots) if t.strip() != '' and p.strip() != '']
        print('Done.')
        train_text = texts[:int(len(texts) * 0.9)]
        val_text = texts[int(len(texts) * 0.9):int(len(texts) * 0.95)]
        test_text = texts[int(len(texts) * 0.95):]

        if make_train:
            train_preproc = Preprocessor_train(tokenizer, train_seq_len, model_type)
            d_train = PlotDataset(train_text, train_preproc)
            print('Train dataset size', len(d_train))
            loaders.append(data.DataLoader(d_train,
                                           # sampler=DistributedSampler(d_train) if distributed else None,
                                           batch_size=train_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=train_collate_fn) if d_train else None)
        if make_val:
            val_preproc = Preprocessor_train(tokenizer, val_seq_len, model_type)
            d_val = PlotDataset(val_text, val_preproc)
            print('Val dataset size', len(d_val))
            loaders.append(data.DataLoader(d_val,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=val_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=val_collate_fn) if d_val else None)
        if make_test:
            test_preproc = Preprocessor_test(tokenizer, test_seq_len, model_type)
            d_test = PlotDataset(test_text, test_preproc, sort=True)
            print('Test dataset size', len(d_test))
            loaders.append(data.DataLoader(d_test,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=test_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=test_collate_fn) if d_test else None)
    else:
        raise Exception('Invalid dataset')

    return loaders


if __name__ == "__main__":
    paragraph_segmentation()

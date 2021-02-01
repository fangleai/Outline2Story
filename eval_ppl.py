import pickle
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from tqdm import tqdm
from tqdm import trange
import importlib
import logging
import copy
from data.util import *


def compute_loss(device, model, input_tokens, target_tokens, mask, loss_fn):
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)

    logits, _ = model(input_tokens)
    num_logits = logits.size(-1)

    # Perform masking
    if mask is not None:
        mask = mask.type(torch.bool)
        mask = mask.to(device)
        logits = logits.masked_select(mask.unsqueeze(-1))
        target_tokens = target_tokens.masked_select(mask)

    ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1))
    loss = ce_loss.float().mean()

    return loss, ce_loss


def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, help='pretrained model path to local checkpoint')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--out-dir', type=str, default='out')

    parser.add_argument('--model_type', type=str, default='b1', choices=['b0', 'b1', 'm'], help="b: baseline, m: model")
    parser.add_argument('--dataset', type=str, default='wp', choices=['wp', 'wi'], help="Dataset to use for training")

    # use GPU
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--no_gpu', action="store_true")

    args = parser.parse_args('--model-path out/wp4.0223/model_latest.pt'.split())
    print(args)

    # GPU
    if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu: torch.cuda.set_device(args.gpu)
    device = torch.device(args.gpu if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed)

    if args.batch_size == -1:
        args.batch_size = 1

    # logging
    save_folder = args.model_path + '.eval/'
    os.makedirs(save_folder, exist_ok=True)
    importlib.reload(logging)
    logging.basicConfig(filename=os.path.join(save_folder, 'eval_ppl.log'),
                        level=logging.INFO, format='%(asctime)s--- %(message)s')
    logging.info('\n----------------------------------------------------------------------')
    #logging.info("the configuration:")
    #logging.info(str(args).replace(',', '\n'))

    print('Loading models...')
    cache_dir = os.path.join(args.out_dir, 'model_cache')
    os.makedirs(cache_dir, exist_ok=True)
    # Load pre-trained teacher tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    tokenizer.max_len = int(1e12)
    model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir)
    # add special tokens
    special_tokens_dict = {
        'pad_token': '<|startoftext|>',
        'cls_token': '<|startofcond|>',
        'sep_token': '<|sepofcond|>',
        'mask_token': '<|endofcond|>'
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'special tokens')
    # Notice: resize_token_embeddings expect to receive the full size of the new vocab
    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer.pad_token == '<|startoftext|>'
    if args.model_path:
        state = torch.load(args.model_path, map_location='cpu')
        if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
            state_copy = copy.copy(state)
            keys = state_copy.keys()
            for k in keys:
                state[k.replace('module.', '')] = state.pop(k)
        model.load_state_dict(state)
        logging.info('load model from ' + args.model_path)
    model.to(device)
    model.eval()  # be careful about model.eval() vs model.train()
    print('Model loaded.')
    ce_loss_fn = nn.CrossEntropyLoss(reduction='none')

    seq_len = model.config.n_ctx
    test_loader = prepare_dataset(
        args.data_dir, args.dataset, tokenizer,
        1, seq_len, 1, seq_len, args.batch_size, seq_len,
        make_train=False, make_val=False, make_test=True, model_type=args.model_type
    )[0]

    logging.info('\n----------------------------------------------------------------------')
    logging.info("Testing loop. batches: %d" % len(test_loader))

    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    startofcond = tokenizer.convert_tokens_to_ids("<|startofcond|>")
    endofcond = tokenizer.convert_tokens_to_ids("<|endofcond|>")

    n_words_bpe = 0
    n_words = 0
    logp_sum = 0.0

    n_words_bpe_raw = 0
    n_words_raw = 0
    logp_sum_raw = 0.0

    with tqdm(total=len(test_loader)) as pbar:
        for i_test, (context, context_mask, keys, storys) in enumerate(test_loader):
            # test_iter = iter(test_loader); context, context_mask, keys, storys = next(test_iter)
            if args.model_type == 'm':
                # our method
                tokens = context.squeeze(0).tolist() + storys[0][0] + [t for k, s in zip(keys[0], storys[0][1:]) for t in
                                                                   [startofcond] + k + s] + [endoftext]
            else:
                # baseline 1, gpt2; baseline 2, gpt2 ablation; tokenizer.decode([198]) = '\n'
                tokens = context.squeeze(0).tolist() + [t for s in storys[0][:-1] for t in s + [198, 198]] + storys[0][-1] + [endoftext]

            if len(tokens) > model.config.n_ctx:
                tokens = tokens[:model.config.n_ctx]
            input_tokens = torch.tensor(tokens[:-1], dtype=torch.long, device=device)
            target_tokens = torch.tensor(tokens[1:], dtype=torch.long, device=device)
            _, ce_loss = compute_loss(device, model, input_tokens, target_tokens, None, ce_loss_fn)

            text = tokens[1:]
            logprob = ce_loss.tolist()
            assert len(text) == len(logprob)

            # raw for all text
            text_story = tokenizer.decode(text).replace('<|endoftext|>', '.').replace('<|startofcond|>', '.').replace('<|sepofcond|>', '.').replace('<|endofcond|>', '.')
            n_words_bpe_raw += len(logprob)
            n_words_raw += len([t for t in re.split('("|\'|!|\?|\.|,|:| |\n|’|“|”|;|\(|\)|`)', text_story) if t != ' ' and t != ''])
            logp_sum_raw += sum(logprob)

            # only for story
            idx = text.index(endoftext)
            text = text[idx + 1:]
            logprob = logprob[idx + 1:]

            if endoftext in text:
                idx = text.index(endoftext)
                text = text[:idx]
                logprob = logprob[:idx]

            assert len(text) == len(logprob)
            story = []
            story_prob = []
            while startofcond in text and endofcond in text and text.index(startofcond) < text.index(endofcond):
                idx = text.index(startofcond)
                story.append(text[:idx])
                story_prob.append(logprob[:idx])
                idx = text.index(endofcond)
                text = text[idx + 1:]
                logprob = logprob[idx + 1:]
            assert len(text) == len(logprob)
            if startofcond not in text and endofcond not in text:
                story.append(text)
                story_prob.append(logprob)
            assert len(story) == len(story_prob)
            logprob_story = [p for lp in story_prob for p in lp]
            text_story = ''.join([tokenizer.decode(s) for s in story]).strip()

            n_words_bpe += len(logprob_story)
            n_words += len([t for t in re.split('("|\'|!|\?|\.|,|:| |\n|’|“|”|;|\(|\)|`)', text_story) if t != ' ' and t != ''])
            logp_sum += sum(logprob_story)

            #logging.info('test sample %05d finished.', i_test)
            pbar.update(1)

    print('Test complete with %05d samples.' % len(test_loader))
    logging.info("Test complete with %05d samples.", len(test_loader))

    ppl_bpe = round(math.exp(logp_sum / n_words_bpe), 3)
    ppl_word = round(math.exp(logp_sum / n_words), 3)
    print(' ppl_word:', ppl_word)
    print(' ppl_bpe :', ppl_bpe)
    logging.info('logp_sum: %f', logp_sum)
    logging.info('n_words_bpe: %d', n_words_bpe)
    logging.info('n_words    : %d', n_words)
    logging.info('    ppl_bpe : %f', ppl_bpe)
    logging.info('    ppl_word: %f', ppl_word)

    ppl_bpe_raw = round(math.exp(logp_sum_raw / n_words_bpe_raw), 3)
    ppl_word_raw = round(math.exp(logp_sum_raw / n_words_raw), 3)
    print(' ppl_word_raw:', ppl_word_raw)
    print(' ppl_bpe_raw :', ppl_bpe_raw)
    logging.info('logp_sum_raw: %f', logp_sum_raw)
    logging.info('n_words_bpe_raw: %d', n_words_bpe_raw)
    logging.info('n_words_raw    : %d', n_words_raw)
    logging.info('    ppl_bpe_raw : %f', ppl_bpe_raw)
    logging.info('    ppl_word_raw: %f', ppl_word_raw)


if __name__ == '__main__':
    run_model()

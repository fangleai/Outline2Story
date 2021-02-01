import pickle
import os
import math
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
import numpy as np
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from tqdm import tqdm
from tqdm import trange
import importlib
import logging
import copy
from data.util import *
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge


def top_k_top_p_filtering(logits, top_k=100, top_p=0.95, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def repeat_score(text, ngram=[3, 4, 5, 6]):
    ngram_list = []
    for ng in ngram:
        ngram_list.append([text[idx:idx + ng] for idx in range(len(text) - ng - 1)])

    max_occurs = []
    for ngrams in ngram_list:
        count_result = Counter([' '.join(n) for n in ngrams])
        try:
            max_occurs.append(
                max(count_result.values())
            )
        except:
            pass

    scores = [max_oc / ((len(text) / ngram[idx]) + ngram[idx]) for idx, max_oc in enumerate(max_occurs)]
    return max(scores) if len(scores) >= 1 else 1.0


def sample_sequence(model, tokenizer, length, batch_size=None, context=None, context_mask=None, temperature=1,
                    top_k=100, top_p=0.95, device='cuda', sample=True, eos_token=None, keys=None):
    assert context is not None
    context = context.to(device)
    context_mask = context_mask.to(device)

    if context.size(1) > 1:
        _, mem = model(context[:, :-1], past=None, attention_mask=context_mask)
        prev = context[:, -1].view(batch_size, -1)
    else:
        mem = None
        prev = context

    kw_idx = torch.tensor([-1] * batch_size, dtype=torch.long, device=device) # use which cond
    kw_w_idx = torch.tensor([-1] * batch_size, dtype=torch.long, device=device) # use which id in the cond
    kw_switch = torch.tensor([False] * batch_size, dtype=torch.bool, device=device) # if *using* cond
    if keys is not None:
        k_len = torch.tensor([len(key) for key in keys], dtype=torch.long, device=device)
        kw_len = [torch.tensor([len(kw) for kw in key] if key else [0], dtype=torch.long, device=device) for key in keys]

    output = context
    probability = torch.tensor([], dtype=torch.float, device=device)
    if_end = torch.tensor([False] * batch_size, dtype=torch.bool, device=device)
    with torch.no_grad():
        for i in range(length): #trange
            logits, mem = model(prev, past=mem)

            logits = logits[:, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            if sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)

            if keys is not None:
                prev = prev.view(-1)
                if prev.eq(tokenizer.convert_tokens_to_ids('<|startofcond|>')).any():
                    switch = prev.eq(tokenizer.convert_tokens_to_ids('<|startofcond|>'))
                    kw_idx[switch] += 1
                    kw_w_idx[switch] = 0
                    kw_switch[switch] = kw_idx[switch] < k_len[switch]

                ls = [l[i].item() for l, i, s in zip(kw_len, kw_idx.tolist(), kw_switch.tolist()) if s] # get lens of kw
                kw_switch[kw_switch] = kw_w_idx[kw_switch] < torch.tensor(ls, dtype=torch.long, device=device)
                next_token = next_token.view(-1)
                next_token[kw_switch] = torch.tensor(
                    [keys[idx][kw_idx[idx]][kw_w_idx[idx]] for idx in range(batch_size) if kw_switch[idx]],
                    dtype=torch.long, device=device)
                next_token = next_token.view(-1, 1)
                kw_w_idx[kw_switch] += 1

            probability = torch.cat((probability, probs.gather(1, next_token)), dim=1)
            output = torch.cat((output, next_token), dim=1)
            prev = next_token

            # early stopping if all sents have ended once
            if_end[next_token.view(-1).eq(eos_token)] = True
            if if_end.all(): break
    return output, probability


def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, help='pretrained model path to local checkpoint')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=int, default=0.95)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--out-dir', type=str, default='out')

    parser.add_argument('--model_type', type=str, default='m', choices=['b0', 'b1', 'm'], help="b: baseline, m: model")
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
    assert args.nsamples % args.batch_size == 0

    # logging
    save_folder = args.model_path + '.eval/'
    os.makedirs(save_folder, exist_ok=True)
    importlib.reload(logging)
    logging.basicConfig(filename=os.path.join(save_folder, 'eval.log'),
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
    model.eval()
    print('Model loaded.')

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

    n_samples = 0
    bleu4_sum = 0.0
    rouge_scores_values_sum = [0.0] * 9

    with tqdm(total=len(test_loader)) as pbar:
        for i_test, (context, context_mask, keys, storys) in enumerate(test_loader):
            # test_iter = iter(test_loader); context, context_mask, keys, storys = next(test_iter)
            if all([len(key)==0 for key in keys]):
                keys = None
            length = args.length
            if length == -1:
                length = model.config.n_ctx - context.size(1)
            elif length > model.config.n_ctx - context.size(1):
                raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

            eff_samples = []
            storys_str = ['\n\n'.join([tokenizer.decode(s) for s in story]) for story in storys] # use '\n\n' as paragraph separator
            for _ in range(args.nsamples // args.batch_size):
                # batch_size, temperature, top_k, top_p, eos_token, sample = args.batch_size, args.temperature, args.top_k, args.top_p, tokenizer.encoder['<|endoftext|>'], True
                out, _ = sample_sequence(
                    model=model,
                    tokenizer=tokenizer,
                    length=length,
                    batch_size=args.batch_size,
                    context=context,
                    context_mask=context_mask,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device = device,
                    eos_token = tokenizer.encoder['<|endoftext|>'],
                    keys=keys
                )
                out = out.tolist()

                # just print
                # generated = 0
                # for i in range(args.batch_size):
                #     generated += 1
                #     text = tokenizer.decode(out[i])
                #     print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                #     print(text)

                # extract story, check metrics
                for i in range(len(out)):
                    text = out[i]
                    text = text[text.index(endoftext) + 1:]

                    if endoftext in text:
                        idx = text.index(endoftext)
                        text = text[:idx]

                    story_sample = []
                    while startofcond in text and endofcond in text and text.index(startofcond) < text.index(endofcond):
                        idx = text.index(startofcond)
                        story_sample.append(text[:idx])
                        idx = text.index(endofcond)
                        text = text[idx + 1:]
                    if startofcond not in text and endofcond not in text:
                        story_sample.append(text)
                    text = '\n\n'.join([tokenizer.decode(s) for s in story_sample]).strip()

                    # score for one long text, higher than 0.075 usually means repetition
                    # rep_score = repeat_score(text.split(), ngram=[3, 4, 5, 6, 7, 8])
                    # if rep_score > 0.075:
                    #     # print(rep_score)
                    #     continue

                    try:
                        # check bleu
                        bleu4 = sentence_bleu([storys_str[i].split()], text, smoothing_function=SmoothingFunction().method7)

                        # check rouge
                        rouge = Rouge()
                        rouge_scores = rouge.get_scores(text, storys_str[i])
                        rouge_scores_values = [v for k in rouge_scores[0].keys() for v in rouge_scores[0][k].values()]

                        bleu4_sum += bleu4
                        rouge_scores_values_sum = [v1 + v2 for v1, v2 in zip(rouge_scores_values_sum, rouge_scores_values)]
                        n_samples += 1
                    except:
                        bleu4 = 0.0
                        rouge_scores = [{'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                                         'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                                         'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}]

                    eff_samples.append((text, bleu4, rouge_scores))

                # write samples to file
                samples_file = open(save_folder + 'batch-' + '%04d' % i_test + '.txt', 'w', encoding='utf8')
                for i in range(len(eff_samples)):
                    samples_file.write("=" * 50 + " SAMPLE " + str(i) + " " + "=" * 50)
                    samples_file.write('\n' * 2)

                    samples_file.write("=" * 40 + " Outlines  " + "=" * 40)
                    samples_file.write('\n' * 2)
                    samples_file.write(tokenizer.decode(context[i, :-1][context_mask[i, :] == 1].tolist()))
                    if keys is not None:
                        samples_file.write('\n\n'.join([tokenizer.decode(s) for s in keys[i]]))
                    samples_file.write('\n' * 2)
                    samples_file.write("=" * 40 + " Story " + "=" * 40)
                    samples_file.write('\n' * 2)
                    samples_file.write(storys_str[i])
                    samples_file.write('\n' * 2)

                    samples_file.write("=" * 40 + " Generated " + "=" * 40)
                    samples_file.write('\n' * 2)
                    samples_file.write(eff_samples[i][0])
                    samples_file.write('\n' * 2)
                    samples_file.write(str(eff_samples[i][1:]))
                    samples_file.write('\n' * 4)
                    samples_file.flush()

                logging.info('batch %04d finished.', i_test)
                pbar.update(1)

    print('Test complete with %05d samples.' % n_samples)
    logging.info("Test complete with %05d samples.", n_samples)

    bleu4 = round(bleu4_sum / n_samples, 3)
    rouge_scores_values = [round(r / n_samples, 3) for r in rouge_scores_values_sum]
    print(' bleu-4:', bleu4)
    print(' rouge :', rouge_scores_values)
    logging.info(' bleu-4: %f', bleu4)
    logging.info(' rouge : %s', str(rouge_scores_values))


if __name__ == '__main__':
    run_model()

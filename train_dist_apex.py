# use DistributedDataParallel instead of DataParallel for data parallelism
import os, time, gc, json, pickle, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn import DataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, logger, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from tqdm import tqdm
import importlib
import logging

from apex.optimizers import FusedAdam
from apex import amp
from apex.fp16_utils import FP16_Optimizer

from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel

from data.util import *
from util import *
from dist_utils import *

devices = '2,1,0'
os.environ["CUDA_VISIBLE_DEVICES"] = devices


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

    ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1)).float().mean()
    loss = ce_loss

    return loss, ce_loss


def train_step(device, model, optimizer, input_tokens, target_tokens, mask, loss_fn):
    loss, ce_loss = compute_loss(device, model, input_tokens, target_tokens, mask, loss_fn)

    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()

    return loss.item(), ce_loss.item()


def main_worker(gpu, ngpus_per_node, args):
    # GPU
    args.gpu = gpu
    print("There are ", torch.cuda.device_count(), " available GPUs!")
    # print('Setting GPUs {}'.format(args.device))
    print('Using GPU devices {}'.format(devices))
    device = torch.device('cuda', args.gpu)
    torch.cuda.set_device(device)
    print('Current single GPU: {}'.format(torch.cuda.current_device()))

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # For multiprocessing distributed training, rank needs to be the global rank among all the processes
    args.rank = args.rank * ngpus_per_node + gpu
    print('Setting rank', args.rank)
    recon_attempt = 1
    connected = False
    if args.rank != 0:
        # Stall to have rank 0 node go first
        time.sleep(3)
    while not connected:
        try:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
            connected = True
            print('Established connection. Rank:', args.rank)
        except Exception as e:
            # Sometimes the head node launches after the worker, which would cause an issue
            print('Failed to init process group. Retrying...', recon_attempt, e)
            recon_attempt += 1
            time.sleep(10)

    # logging
    if args.rank == 0:
        save_folder = os.path.join(args.out_dir, args.experiment)
        os.makedirs(save_folder, exist_ok=True)
        t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
        v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)
        importlib.reload(logging)
        logging.basicConfig(filename=os.path.join(save_folder, 'train.log'),
                            level=logging.INFO, format='%(asctime)s--- %(message)s')
        logging.info('\n*******************************************************************************\n')
        logging.info("the configuration:")
        logging.info(str(args).replace(',', '\n'))

    print('Loading models...')
    cache_dir = os.path.join(args.out_dir, 'model_cache')
    os.makedirs(cache_dir, exist_ok=True)
    # Load pre-trained teacher tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    # Hack to allow tokenizing longer sequences.
    tokenizer.max_len = int(1e12)
    model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir)
    if args.load:
        if args.load == 'none':
            print('Randomly initializing model weights...')
            model.apply(model.init_weights)
        else:
            print('Loading model weights...')
            model.load_state_dict(torch.load(os.path.join(args.load, 'model_latest.pt'), map_location='cpu'))
            gc.collect()
    print('params:', num_params(model))  # gpt2: 124439808
    print('Done.')

    print('Setup data...')
    # Batch and sequence length schedule
    assert len(args.batch_sizes) == len(args.seq_lens)
    batch_schedule = list(zip(map(int, args.batch_sizes), map(int, args.seq_lens)))
    assert len(batch_schedule) == 2, 'Currently not supporting multiple schedule'
    cur_b_schedule = len(batch_schedule) - 1 if args.switch_time == 0 else 0

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

    print('Batch schedule', batch_schedule)
    train_loader, val_loader = prepare_dataset(
        args.data_dir, args.dataset, tokenizer,
        batch_schedule[cur_b_schedule][0], batch_schedule[cur_b_schedule][1],
        batch_schedule[-1][0], batch_schedule[-1][1],
        num_workers=args.workers, model_type=args.model_type
    )
    print('Done.')

    print('Wrapping models and optimizers...')
    # Apply linear scaling rule to increase batch size for short sequence training.
    lr_schedule = switch_schedule(linear_schedule(args), batch_schedule[cur_b_schedule][0] / batch_schedule[-1][0],
                                  int(args.iterations * args.switch_time))
    model.train()
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O3')
    loss_model = DDP(model)

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    print('Done.')

    print('Begin training iterations')
    logging.info("Begin training iterations")
    max_val_batches = 1000  # max num. of val batches
    save_interval = 50000  # num. of inter to save a model
    logging.info("Total iteration: %d" % args.iterations)
    e = 0  # number of epoch
    num_iters = 0
    optimizer.zero_grad()

    def val_step(val_loader):
        with torch.no_grad():
            logging.info("Validation loop.         Batches: %d" % len(val_loader))
            logging.info("Validation loop. max_val_batches: %d" % max_val_batches)
            stats = []
            # Validation
            # input_tokens, target_tokens, mask = next(iter(val_loader))
            with tqdm(total=min(len(val_loader), max_val_batches)) as pbar:
                for i, (input_tokens, target_tokens, mask) in enumerate(val_loader):
                    loss, ce_loss = compute_loss(device, model, input_tokens, target_tokens, mask, loss_fn)
                    stats.append([loss.item(), math.exp(ce_loss.item())])

                    if i > max_val_batches:
                        break
                    pbar.update(1)

            stats = np.mean(stats, axis=0)
            v_writer.add_scalar('loss', stats[0], num_iters)
            v_writer.add_scalar('ppl', stats[1], num_iters)
            logging.info('val loss: %.4f' % stats[0])
            logging.info('val  ppl: %.4f' % stats[1])

    while num_iters < args.iterations:
        # Run epoch
        st = time.time()

        # Training
        print('Training loop. Batches:', len(train_loader))
        logging.info('\n----------------------------------------------------------------------')
        logging.info("Training loop.       Batches: %d" % len(train_loader))
        logging.info("Training loop. save_interval: %d" % save_interval)

        # train_iter = iter(train_loader); input_tokens, target_tokens, mask = next(train_iter)
        with tqdm(total=len(train_loader)) as pbar:
            for i, (input_tokens, target_tokens, mask) in enumerate(train_loader):
                # Normal grad step
                optimizer.zero_grad()
                loss, ce_loss = train_step(device, loss_model, optimizer, input_tokens, target_tokens, mask, loss_fn)
                optimizer.step()

                if args.rank == 0:
                    lr = scheduler.get_last_lr()[0]
                    # Log to Tensorboard
                    t_writer.add_scalar('loss', loss, num_iters)
                    t_writer.add_scalar('ppl', math.exp(ce_loss), num_iters)
                    t_writer.add_scalar('lr', lr, num_iters)
                    t_writer.add_scalar('iter_time', time.time() - st, num_iters)

                st = time.time()
                end = num_iters >= args.iterations

                if args.warmup != -1:
                    scheduler.step()

                if end: break
                num_iters += 1
                pbar.update(1)

                if args.switch_time > 0 and num_iters == int(args.iterations * args.switch_time):
                    print('Switch to long sequence training')
                    logging.info("Switch to long sequence training")
                    cur_b_schedule += 1
                    train_loader, val_loader = prepare_dataset(
                        args.dataset_dir, args.dataset_name, tokenizer,
                        batch_schedule[cur_b_schedule][0], batch_schedule[cur_b_schedule][1],
                        batch_schedule[-1][0], batch_schedule[-1][1]
                    )
        e += 1
        logging.info("Training loop. The ith epoch completed: %d" % e)

        if args.rank == 0:
            print('Saving model...')
            logging.info('\n------------------------------------------------------')
            logging.info("Iteration completed: %d, remained %d" % (num_iters, args.iterations - num_iters))
            logging.info("Saving model...")
            #torch.save(model.state_dict(), os.path.join(save_folder, 'model_{:02d}.pt'.format(num_iters // save_interval)))
            torch.save(model.state_dict(), os.path.join(save_folder, 'model_latest.pt'))
            torch.save(optimizer.state_dict(), os.path.join(save_folder, 'opt_latest.pt'))
            torch.save(scheduler.state_dict(), os.path.join(save_folder, 'scheduler_latest.pt'))
            val_step(val_loader)

    print('Training complete.')
    logging.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)

    # Default parameters are set based on single GPU training
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument('--model_type', type=str, default='m', choices=['b0', 'b1', 'm'], help="b: baseline, m: model")
    parser.add_argument('--iterations', type=int, default=425001)  # num. of batchs to have samples wp 850001 wi 300001
    parser.add_argument('--dataset', type=str, default='wp', choices=['wp', 'wi'], help="Dataset to use for training")
    parser.add_argument('--warmup', type=int, default=5000,
                        help="Amount of iterations to warmup, then decay. (-1 for no warmup and decay)")

    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[2, 1],
                        help='batch size per GPU. Lists the schedule.')
    parser.add_argument('--seq-lens', nargs='+', type=int, default=[512, 1024],
                        help='seq length per sample. Lists the schedule.')
    parser.add_argument('--switch-time', type=float, default=0,
                        help="Percentage of iterations to spend on short sequence training.")
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--out-dir', type=str, default='out')
    parser.add_argument('--load', type=str, help='path to load model from')

    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--fp16', action='store_true', help="Train using FP16?")

    args = parser.parse_args('test --batch-sizes 2 2 --seq-lens 512 1024 --fp16'.split())

    # Each node is expected to have same number of GPUs
    ngpus_per_node = torch.cuda.device_count()
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

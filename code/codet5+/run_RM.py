import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from torch.optim import AdamW
from transformers import (get_scheduler, T5Config, RobertaTokenizer, AutoTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer, T5Model,
                        get_linear_schedule_with_warmup)
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import argparse
import logging

from RewardModel import RewardModel, compute_rank_list_loss, compute_combined_rank_loss
from RewardDataset import RewardCloneDataset as RewardDataset


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'codet5+': (T5Config, T5ForConditionalGeneration, AutoTokenizer)
}


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def reward_dataset_collate_fn(batch):

    batch_input_ids = []
    batch_attention_mask = []
    for item in batch:
            input_ids = torch.stack([x["input_ids"] for x in item])           # (rank_list_len, block_size)
            attention_mask = torch.stack([x["attention_mask"] for x in item]) # (rank_list_len, block_size)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask
    }


def evaluate(args, model, tokenizer):
    eval_output_dir = args.output_dir

    eval_dataset = RewardDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=reward_dataset_collate_fn)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    correct_top1 = 0
    with torch.no_grad():
        all_rank_rewards = []
        for inputs in eval_dataloader:
            batch_rank_rewards = []
            for batch_idx in range(len(inputs['input_ids'])):
                input_ids = inputs['input_ids'][batch_idx].to(args.device)  # [class_num, seq_len]
                attention_mask = inputs['attention_mask'][batch_idx].to(args.device)
                rewards = model(input_ids=input_ids, attention_mask=attention_mask)
                batch_rank_rewards.append(rewards)
                all_rank_rewards.append(rewards)

            loss = compute_combined_rank_loss(batch_rank_rewards, alpha=0.0, device=args.device)
            eval_loss += loss.mean().item()
            nb_eval_steps += 1

    total_ranklist, right_ranklist = 0, 0
    for rank_rewards in all_rank_rewards:
        rank_rewards = [t.cpu().float() for t in rank_rewards]
        rank_rewards_sorted = sorted(rank_rewards, reverse=True)
        total_ranklist += 1
        if rank_rewards_sorted == rank_rewards:
            right_ranklist += 1

        # Top1 -acc
        top1_index = torch.argmax(torch.tensor(rank_rewards))
        if top1_index == 0:
            correct_top1 += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    eval_acc = right_ranklist / total_ranklist
    top1_acc = correct_top1 / total_ranklist
    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        "top1_acc": round(top1_acc, 4)
    }
    return result


def train(args, train_dataset, model, tokenizer):
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=reward_dataset_collate_fn)
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_acc = 0.0
    model.zero_grad()

    # tensorboard
    if os.path.exists(args.runs_path) is False:
        os.makedirs(args.runs_path)
    tb_writer = SummaryWriter(args.runs_path)
    tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        right_ranklist = 0
        total_ranklist = 0
        correct_top1 = 0
        all_rank_rewards = []
        for step, inputs in enumerate(bar):
            batch_rank_rewards = []
            for batch_idx in range(len(inputs['input_ids'])):
                input_ids = inputs['input_ids'][batch_idx].to(args.device)  # [5, seq_len]
                attention_mask = inputs['attention_mask'][batch_idx].to(args.device)
                rewards = model(input_ids=input_ids, attention_mask=attention_mask)
                batch_rank_rewards.append(rewards)
                all_rank_rewards.append(rewards)

            loss = compute_combined_rank_loss(batch_rank_rewards, alpha=0.0, device=args.device)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value, 4))
                            # Save model checkpoint

                    if results['eval_acc'] > best_acc:
                        best_acc = results['eval_acc']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best acc:%s", round(best_acc, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

        for rank_rewards in all_rank_rewards:
            rank_rewards = [t.cpu().float() for t in rank_rewards]
            rank_rewards_sorted = sorted(rank_rewards, reverse=True)
            total_ranklist += 1
            if rank_rewards_sorted == rank_rewards:
                right_ranklist += 1

            # Top1 -acc
            top1_index = torch.argmax(torch.tensor(rank_rewards))
            if top1_index == 0:
                correct_top1 += 1

        # Calculate average loss for the epoch
        avg_loss = train_loss / tr_num
        train_acc = right_ranklist / total_ranklist
        top1_acc = correct_top1 / total_ranklist
        logger.info("  train acc:%s", round(train_acc, 4))
        logger.info("  train top1 acc:%s", round(top1_acc, 4))
        tb_writer.add_scalar(tags[0], avg_loss, idx)
        tb_writer.add_scalar(tags[1], train_acc, idx)
        tb_writer.add_scalar(tags[2], results['eval_loss'], idx)
        tb_writer.add_scalar(tags[3], results['eval_acc'], idx)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], idx)
        tb_writer.add_scalar("train top1 acc", top1_acc, idx)
        tb_writer.add_scalar("val top1 acc", results['top1_acc'], idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file", default='../reward_dataset/BigCloneBench/train.jsonl', type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--eval_data_file", default='../reward_dataset/BigCloneBench/valid.jsonl', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--output_dir", type=str,
                        default='',
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--runs_path', type=str,
                        default='.', help='save runs path')

    parser.add_argument('--seed', type=int, default=123456,
                        help="random seed for initialization")
    parser.add_argument("--model_type", default="codet5+", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default='./code/codet5p-220m', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="./code/codet5p-220m", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument('--epoch', type=int, default=10,
                        help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")

    parser.add_argument("--block_size", default=400, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")

    parser.add_argument("--do_train", default=True, action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=False, action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--evaluate_during_training", default=True, action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # Add early stopping parameters and dropout probability parameters
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("--min_loss_delta", type=float, default=0.001,
                        help="Minimum change in the loss required to qualify as an improvement.")
    parser.add_argument('--dropout_probability', type=float, default=0, help='dropout probability')
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    model = RewardModel(model, config, args)
    logger.info("Training/evaluation parameters %s", args)
    train_dataset = RewardDataset(tokenizer, args, args.train_data_file)
    train(args, train_dataset, model, tokenizer)

if __name__ == "__main__":
    main()

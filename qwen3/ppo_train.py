import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import (AutoTokenizer, get_linear_schedule_with_warmup, Qwen3Config,
                          Qwen3ForSequenceClassification, Qwen3Model)
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
import numpy as np
import random
import json
import logging
import argparse
from tqdm import tqdm
from model import Model  # 引入你的分类模型
from RewardModel import RewardModel  # 引入你之前定义的奖励模型


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'qwen3': (Qwen3Config, Qwen3ForSequenceClassification, Qwen3Model, AutoTokenizer),
}


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 text1_text,
                 text2_text,
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label
        self.text1_text = text1_text
        self.text2_text = text2_text


def convert_examples_to_features(js,tokenizer,args):
    tokenizer.cls_token = '<s>'
    tokenizer.sep_token = '</s>'

    text1 = ' '.join(js['text1'].split())
    text2 = ' '.join(js['text2'].split())
    text1_tokens = tokenizer.tokenize(text1)
    text2_tokens = tokenizer.tokenize(text2)
    max_len = args.block_size - 3
    half = max_len // 2
    text1_tokens = text1_tokens[:half]
    text2_tokens = text2_tokens[:half]
    # --- 拼接：[CLS] code1 [SEP] code2 [SEP] ---
    source_tokens = [tokenizer.cls_token] + text1_tokens + [tokenizer.sep_token] + text2_tokens + [tokenizer.sep_token]

    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, js['label'], text1, text2)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js, tokenizer, args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:1]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format(
                    [x.replace('\u0120', '_') if x is not None else '' for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), self.examples[i].text1_text, self.examples[i].text2_text


class AdaptiveKLController:
    """
    Adaptive KL controller with clipping to prevent exploding or vanishing KL coefficients.
    Based on: https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef=0.1, target=0.1, horizon=10000, min_kl_coef=1e-4, max_kl_coef=1.0):
        """
        Args:
            init_kl_coef (float): Initial KL coefficient.
            target (float): Target KL divergence.
            horizon (int): Number of steps over which to adjust.
            min_kl_coef (float): Lower bound for KL coefficient.
            max_kl_coef (float): Upper bound for KL coefficient.
        """
        self.kl_coef = init_kl_coef
        self.target = target
        self.horizon = horizon
        self.min_kl_coef = min_kl_coef
        self.max_kl_coef = max_kl_coef

    def update(self, current_kl, n_steps=1):
        """
        Update KL coefficient based on current KL.

        Args:
            current_kl (float): Observed KL divergence.
            n_steps (int): Number of training steps since last update.
        """
        proportional_error = np.clip(current_kl / self.target - 1, -0.2, 0.2)
        multiplier = 1.0 + proportional_error * n_steps / self.horizon
        self.kl_coef *= multiplier
        self.kl_coef = float(np.clip(self.kl_coef, self.min_kl_coef, self.max_kl_coef))


# 设置随机种子确保可复现
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(args, model, tokenizer):
    eval_output_dir = args.output_dir
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1,
                                 pin_memory=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit, _ = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = np.argmax(logits, axis=1)
    eval_mcc = matthews_corrcoef(labels, preds)
    eval_acc = np.mean(labels == preds)
    eval_f1 = f1_score(labels, preds, average='macro')
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        "eval_f1": round(eval_f1, 4),
        "eval_mcc": round(eval_mcc, 4)
    }
    return result


def compute_rewards(args, inputs, predictions, reward_model, tokenizer, label_map):
    """
    根据原始inputs和模型预测，构建 reward model 输入并计算奖励值。

    Args:
        inputs (torch.Tensor): 原始 input_ids (batch_size, seq_len)
        predictions (torch.Tensor): 当前模型预测的类别 (batch_size,) — 取值为0, 1, 2 ...
        reward_model (nn.Module): 奖励模型
        tokenizer (transformers.PreTrainedTokenizer): 分词器

    Returns:
        rewards (torch.Tensor): 奖励分数，shape (batch_size,)
    """
    # 将input_ids decode成文本
    texts = tokenizer.batch_decode(inputs, skip_special_tokens=True)
    # 把预测结果 verbalize
    def verbalize(pred, label_map):
        label_str = label_map.get(str(pred), f"class {pred}")
        return f"Prediction: the sentence is {label_str}"

    predictions_text = [verbalize(pred.item(), label_map) for pred in predictions]

    # 构建新的输入
    prefix = "sentence: "
    middle = " || "
    new_inputs = []
    for text, pred_text in zip(texts, predictions_text):

        # 重新分词，注意截断
        pred_tokens = tokenizer.tokenize(middle + pred_text)

        pred_ids = tokenizer.convert_tokens_to_ids(pred_tokens)

        max_text_len = args.block_size - len(pred_ids) - 2  # [CLS] 和 [SEP]
        text_tokens = tokenizer.tokenize(prefix + text)[:max_text_len]

        text_ids = tokenizer.convert_tokens_to_ids(text_tokens)

        tokenizer.cls_token = '<s>'
        tokenizer.sep_token = '</s>'
        input_ids = [tokenizer.cls_token_id] + text_ids + pred_ids + [tokenizer.sep_token_id]

        attention_mask = [1] * len(input_ids)

        # padding
        padding_length = args.block_size - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length

        new_inputs.append({
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask)
        })

    # 将batch合并
    input_ids = torch.stack([x["input_ids"] for x in new_inputs]).to(inputs.device)
    attention_mask = torch.stack([x["attention_mask"] for x in new_inputs]).to(inputs.device)

    with torch.no_grad():
        rewards = reward_model(input_ids=input_ids, attention_mask=attention_mask)

    # return torch.tanh(rewards)
    return rewards


def compute_rewards_pair(args, text1, text2, predictions, reward_model, tokenizer, label_map):
    # 把预测结果 verbalize
    def verbalize(pred, label_map):
        label_str = label_map.get(str(pred), f"class {pred}")
        return f"Prediction: the two sentences are {label_str}"

    predictions_text = [verbalize(pred.item(), label_map) for pred in predictions]

    # 构建新的输入
    prefix = "sentence: "
    middle = " || "
    new_inputs = []
    tokenizer.cls_token = '<s>'
    tokenizer.sep_token = '</s>'
    for t1, t2, pred_text in zip(text1, text2, predictions_text):
        # 重新分词，注意截断
        pred_tokens = tokenizer.tokenize(middle + pred_text)
        pred_ids = tokenizer.convert_tokens_to_ids(pred_tokens)
        max_text_len = args.block_size - len(pred_ids) - 3  # [CLS] 和 [SEP]
        half = max_text_len // 2
        text1_tokens = tokenizer.tokenize(prefix + t1)[:half]
        text2_tokens = tokenizer.tokenize(prefix + t2)[:half]
        text1_ids = tokenizer.convert_tokens_to_ids(text1_tokens)
        text2_ids = tokenizer.convert_tokens_to_ids(text2_tokens)
        input_ids = [tokenizer.cls_token_id] + text1_ids + [tokenizer.sep_token_id] + text2_ids + pred_ids + [
            tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        # padding
        padding_length = args.block_size - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length

        new_inputs.append({
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask)
        })

    # 将batch合并
    input_ids = torch.stack([x["input_ids"] for x in new_inputs]).to(args.device)
    attention_mask = torch.stack([x["attention_mask"] for x in new_inputs]).to(args.device)

    with torch.no_grad():
        rewards = reward_model(input_ids=input_ids, attention_mask=attention_mask)

    # return torch.tanh(rewards)
    return rewards


def train(args, train_dataset, model, ref_model, reward_model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=1, pin_memory=True)
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch

    model.to(args.device)
    ref_model.to(args.device)
    reward_model.to(args.device)
    reward_model.eval()
    ref_model.eval()

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    kl_controller = AdaptiveKLController(init_kl_coef=0.1, target=0.1, horizon=10000)
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
    best_f1 = 0.0
    best_acc = 0.0
    best_mcc = 0.0
    model.zero_grad()

    tr_kl_mean, tr_kl_coef, tr_reward, tr_ad_reward = 0.0, 0.0, 0.0, 0.0

    # tensorboard
    if os.path.exists(args.runs_path) is False:
        os.makedirs(args.runs_path)
    tb_writer = SummaryWriter(args.runs_path)

    json_label_path = args.json_path
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    with open(json_label_path, "r") as f:
        label_map = json.load(f)

    step_train = 0
    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()

            with torch.no_grad():
                # Reference Model推理
                ref_probs, _ = ref_model(inputs)
                old_log_probs = torch.log(ref_probs + 1e-8)  # 防止log(0)

            # 策略模型推理
            new_probs, values = model(inputs)
            # 这里的multinomial可以改成actions = torch.multinomial(new_probs, num_samples=1).squeeze(-1)
            actions = torch.argmax(new_probs, dim=-1)
            log_probs = torch.log(new_probs + 1e-8)
            action_log_probs = log_probs[range(len(actions)), actions]
            old_action_log_probs = old_log_probs[range(len(actions)), actions]

            # KL散度计算
            kl = torch.sum(new_probs * (log_probs - old_log_probs), dim=-1)
            kl_mean = kl.mean().item()

            # 奖励模型打分
            rewards = compute_rewards_pair(args, batch[2], batch[3], actions, reward_model, tokenizer, label_map)
            adjusted_rewards = rewards - kl_controller.kl_coef * kl

            # Advantage 引入 baseline 值函数
            if args.vf_coef == 0.0:
                advantage = adjusted_rewards.detach()
            else:
                advantage = adjusted_rewards - values.detach()

            adv_std = advantage.std()
            if torch.isnan(advantage).any() or torch.isnan(adv_std) or advantage.std() < 1e-8:
                adv_std = 1.0
                print(
                    f"[Debug] step={step} | adv_mean={advantage.mean().item():.4f} | adv_std={advantage.std().item():.6f} | reward_mean={rewards.mean().item():.4f}")
                advantage = (advantage - advantage.mean()) / adv_std
            else:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # PPO 损失
            ratio = torch.exp(action_log_probs - old_action_log_probs)
            surrogate1 = ratio * advantage
            surrogate2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * advantage
            ppo_loss = -torch.mean(torch.min(surrogate1, surrogate2))

            # 值函数损失
            value_loss = F.mse_loss(values, adjusted_rewards.detach())
            loss = ppo_loss + args.vf_coef * value_loss  # args.vf_coef ≈ 0.5 通常

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
            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            # 记录PPO过程的变化
            tr_kl_mean += kl_mean
            tr_kl_coef += kl_controller.kl_coef
            tr_reward += rewards.mean().item()
            tr_ad_reward += adjusted_rewards.mean().item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                kl_controller.update(kl_mean, n_steps=1)

                tb_writer.add_scalar("Loss/policy", loss.item(), step_train)
                tb_writer.add_scalar("KL/value", kl_mean, step_train)
                tb_writer.add_scalar("KL/coef", kl_controller.kl_coef, step_train)
                tb_writer.add_scalar("Reward/mean", rewards.mean().item(), step_train)
                tb_writer.add_scalar("Adjusted_Reward/mean", adjusted_rewards.mean().item(), step_train)
                tb_writer.add_scalar("advantage/mean", advantage.mean().item(), step_train)

                step_train += 1

                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and (step + 1) % args.save_steps == 0:

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

        # Calculate average loss for the epoch
        avg_loss = train_loss / tr_num
        tb_writer.add_scalar("Loss", avg_loss, idx)
        tb_writer.add_scalar("KL/value-1", tr_kl_mean / tr_num, idx)
        tb_writer.add_scalar("KL/coef-1", tr_kl_coef / tr_num, idx)
        tb_writer.add_scalar("Reward/mean-1", tr_reward / tr_num, idx)
        tb_writer.add_scalar("Adjusted_Reward/mean-1", tr_ad_reward / tr_num, idx)
        tb_writer.add_scalar("val_loss", results['eval_loss'], idx)
        tb_writer.add_scalar("val_acc", results['eval_acc'], idx)
        tb_writer.add_scalar("val_f1", results['eval_f1'], idx)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_labels', type=int, default=2,
                        help='class num')
    parser.add_argument("--json_path", default='../MRPC.json', type=str, )
    parser.add_argument("--train_data_file", default='../dataset/MRPC/train.jsonl', type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--eval_data_file", default='../dataset/MRPC/valid.jsonl', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--sft_path", default='./weights/three/MRPC/CE-lr0.00002-seed1', type=str,
                        help="")
    parser.add_argument("--reward_path", default='./weights/three/MRPC/Reward-ccrl-alpha0.0-lr0.00001-seed1', type=str,
                        help="")
    parser.add_argument("--output_dir", default='./weights/three/MRPC/PPO-ccrl-alpha0.0-vf0.0-lr0.000001-seed1', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--runs_path', type=str,
                        default='./runs/three/MRPC/PPO-ccrl-alpha0.0-vf0.0-lr0.000001-seed1', help='save runs path')
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for initialization")

    parser.add_argument('--clip_range', type=float, default=0.2,
                        help="")
    parser.add_argument('--vf_coef', type=float, default=0.0,
                        help="Value function loss coefficient.")

    parser.add_argument("--learning_rate", default=1e-6, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--model_type", default="qwen3", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default='./pre', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="./pre", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument('--epoch', type=int, default=10,
                        help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

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
    if args.n_gpu == 0:
        args.n_gpu = 1
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

    config_class, model_class, reward_model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
        reference_model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
        reward_model = reward_model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)
        reference_model = model_class(config)
        reward_model = reward_model_class(config)
    model.config.pad_token_id = tokenizer.pad_token_id
    reference_model.config.pad_token_id = tokenizer.pad_token_id

    model = Model(model, config, tokenizer, args)
    model.load_state_dict(torch.load(os.path.join(args.sft_path, "checkpoint-best-acc/model.bin"), map_location=device))
    reference_model = Model(reference_model, config, tokenizer, args)
    reference_model.load_state_dict(torch.load(os.path.join(args.sft_path, "checkpoint-best-acc/model.bin"), map_location=device))
    for p in reference_model.parameters():
        p.requires_grad = False
    reference_model.eval()

    reward_model = RewardModel(reward_model)
    reward_model.load_state_dict(torch.load(os.path.join(args.reward_path, "checkpoint-best-acc/model.bin"), map_location=device))
    reward_model.eval()

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        if args.local_rank == 0:
            torch.distributed.barrier()

        train(args, train_dataset, model, reference_model, reward_model, tokenizer)


if __name__ == "__main__":
    main()
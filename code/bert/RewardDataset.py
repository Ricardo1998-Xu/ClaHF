from torch.utils.data import Dataset
import json
import torch


class RewardDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.data = []
        self.tokenizer = tokenizer
        self.block_size = args.block_size

        with open(file_path, "r", encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = ' '.join(item["text"].split())
        candidates = item["candidates"]
        inputs = []
        for cand in candidates:
            input = self.build_input(text, cand)
            inputs.append(input)

        return inputs

    def build_input(self, text, prediction):

        middle = " || "

        pred_tokens = self.tokenizer.tokenize(middle + prediction)
        pred_ids = self.tokenizer.convert_tokens_to_ids(pred_tokens)

        max_code_len = self.block_size - len(pred_ids) - 2  # [CLS] 和 [SEP]

        code_tokens = self.tokenizer.tokenize(text)[:max_code_len]

        code_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)

        input_ids = [self.tokenizer.cls_token_id] + code_ids + pred_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        # padding
        padding_length = self.block_size - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask)
        }


class RewardPairDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.data = []
        self.tokenizer = tokenizer
        self.block_size = args.block_size

        with open(file_path, "r", encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text1 = ' '.join(item["text1"].split())
        text2 = ' '.join(item["text2"].split())
        candidates = item["candidates"]
        inputs = []
        for cand in candidates:
            input = self.build_input(text1, text2, cand)
            inputs.append(input)
        return inputs

    def build_input(self, text1, text2, prediction):
        # [CLS] text1 [SEP] text2 || prediction [SEP] ---

        middle = " || "

        pred_tokens = self.tokenizer.tokenize(middle + prediction)
        pred_ids = self.tokenizer.convert_tokens_to_ids(pred_tokens)

        max_code_len = self.block_size - len(pred_ids) - 3  # [CLS] 和 [SEP]
        half = max_code_len // 2

        text1_tokens = self.tokenizer.tokenize(text1)[:half]
        text2_tokens = self.tokenizer.tokenize(text2)[:half]
        text1_ids = self.tokenizer.convert_tokens_to_ids(text1_tokens)
        text2_ids = self.tokenizer.convert_tokens_to_ids(text2_tokens)

        input_ids = [self.tokenizer.cls_token_id] + text1_ids + [self.tokenizer.sep_token_id] + text2_ids + pred_ids + [
            self.tokenizer.sep_token_id]

        attention_mask = [1] * len(input_ids)

        # padding
        padding_length = self.block_size - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask)
        }

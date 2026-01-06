# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss


class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.classifier = nn.Linear(self.encoder.config.hidden_size, args.num_labels)
        self.value_head = nn.Linear(self.encoder.config.hidden_size, 1)
        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)

        
    def forward(self, input_ids=None,labels=None):
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        hidden_states = self.dropout(hidden_states)

        last_token_logits = hidden_states[:, -1, :]  # (batch_size, hidden_size)

        logits = self.classifier(last_token_logits)  # (batch_size, 2)
        values = self.value_head(last_token_logits).squeeze(-1)
        prob = nn.functional.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob, values
        else:
            return prob, values
      
        
 

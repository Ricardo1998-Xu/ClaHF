# ClaHF: A Human Feedback-inspired Reinforcement Learning Framework for Improving Classification Tasks

## 🚀 Overview
**ClaHF**: a human feedback-inspired reinforcement learning framework for text classification that integrates preference modeling and RL optimization into the classification pipeline without requiring additional human annotations. The overall framework of ClaHF is illustrated in Figure. (a) SFT to provide high-quality initialization. (b) Automatic construction of preference data from the original classification dataset. (c) Training the RM with preference data. (d) RL optimization of the policy model using the trained RM.

![image](figure/fig2.png?raw=true)

This repository provides an end-to-end implementation of ClaHF, including:
- Supervised Fine-Tuning (SFT)
- Reward Model training with pairwise preferences
- PPO-based optimization for classification models
- Adaptive KL control and evaluation on multiple datasets
  
---

## 📂 Repository Structure
```bash
ClaHF/
│── 📁 Dataset/                     # Contains datasets used in the study
│   ├── 📂 CoLA/                    # The CoLA dataset
│   │   ├── 📜 train.jsonl          # Training dataset
│   │   ├── 📜 test.jsonl           # Testing dataset
│   │   ├── 📜 valid.jsonl          # valid dataset
│   │   └── ...
│   ├── 📂 MRPC/
│   ├── 📂 SST-5/              
│   └── ...               
│
│── 📁 Pre_Dataset/                 # Preference datasets
│   ├── 📂 CoLA/                    # The CoLA dataset
│   │   ├── 📜 train.jsonl          # Training dataset
│   │   ├── 📜 test.jsonl           # Testing dataset
│   │   ├── 📜 valid.jsonl          # valid dataset
│   │   └── ...
│   ├── 📂 MRPC/
│   ├── 📂 SST-5/              
│   └── ...                    
│
│── 📁 Code/            # Implementations of classification models
│   ├── 🤖 Test1_bert/              # BERT model implementation
│   │   ├── 📜 clss_indices.json    # Label mapping file
│   │   ├── 📜 model.py             # Model definition
│   │   ├── 📜 RewardModel.py       # Reward model definition
│   │   ├── 📜 run.py               # Script for fine-tuning the model
│   │   ├── 📜 run_RL.py            # RL optimization
│   │   ├── 📜 run_RM.py            # Script for training the RM
│   │   ├── 📜 test.py              # Script for model evaluation
│   │   └── ...
│   ├── 🤖 Test2_codebert/          # CodeBERT model implementation
│   ├── 🤖 Test3_t5/                # T5 model implementation
│   ├── 🤖 Test4_codet5/            # CodeT5 model implementation
│   ├── 🤖 Test5_codet5+/           # CodeT5+ model implementation
│   ├── 🤖 Test6_opt/               # OPT model implementation
│   ├── 🤖 Test7_codegen/           # CodeGen model implementation
│   └── 🤖 Test8_qwen3/             # QWen3 model implementation
│   
│── 📜 environment.yaml             # Environment configuration file
│── 📜 README.md                    
└── ...
```

---

## 💻 Experiments
### 📥 Install
```sh
conda env create -f environment.yml
```

### 🚀 Training Pipeline

#### Step 1: Supervised Fine-Tuning (SFT)
Train a base classifier with labeled data.
```sh
python run.py \
    --num_labels=5 \
    --train_data_file=. \
    --eval_data_file=. \
    --output_dir=./saved_models \
    --runs_path=./runs \
    --model_type=qwen3 \
    --tokenizer_name=Qwen/Qwen3-0.6B \
    --model_name_or_path=Qwen/Qwen3-0.6B \
    --do_train \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 1
    --adam_epsilon 1e-8
    --evaluate_during_training \
    --seed 123456
```

#### Step 2: Reward Model Training
Then train the reward model with Top-1 + Pairwise Loss:
```sh
python run_RM.py \
    --train_data_file=. \
    --eval_data_file=. \
    --output_dir=./saved_models \
    --runs_path=./runs \
    --model_type=qwen3 \
    --tokenizer_name=Qwen/Qwen3-0.6B \
    --model_name_or_path=Qwen/Qwen3-0.6B \
    --do_train \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 1
    --adam_epsilon 1e-8
    --evaluate_during_training \
    --seed 123456
```

#### Step 3: PPO Optimization
Use the SFT model as policy initialization and optimize with reward feedback.
```sh
python run_RL.py \
    --num_labels=5 \
    --json_path=./SST-5.json \
    --train_data_file=. \
    --eval_data_file=. \
    --sft_path=checkpoints/sft \
    --reward_path=checkpoints/reward \
    --output_dir=./saved_models \
    --runs_path=./runs \
    --model_type=qwen3 \
    --tokenizer_name=Qwen/Qwen3-0.6B \
    --model_name_or_path=Qwen/Qwen3-0.6B \
    --do_train \
    --epoch 10 \
    --clip_range 0.2 \
    --vf_coef 0.25
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-6 \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 1
    --adam_epsilon 1e-8
    --evaluate_during_training \
    --seed 123456
```

#### Example: Evaluation
```sh
python test.py \
    --test_data_file=. \
    --output_dir=./saved_models \
    --results_path=./results \
    --model_type=qwen3 \
    --tokenizer_name=Qwen/Qwen3-0.6B \
    --model_name_or_path=Qwen/Qwen3-0.6B \
    --do_test \
    --block_size 400 \
    --eval_batch_size 8 \
    --seed 123456
```
Metrics include:
Accuracy, F1, Expected Calibration Error (ECE), MCC

---

## 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

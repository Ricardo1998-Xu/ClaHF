# ClaHF: A Human Feedback-inspired Reinforcement Learning Framework for Improving Classification Tasks

## 🚀 Overview
**ClaHF**: a human feedback-inspired reinforcement learning (RL) framework for text classification that integrates preference modeling and RL optimization into the classification pipeline without requiring additional human annotations.
---

## 📂 Repository Structure
```bash
BioDefect/
│── 📁 Dataset/                     # Contains datasets used in the study, including BioDefect
│   ├── 📂 BioDefect/               # The BioDefect dataset
│   │   ├── 📜 train.jsonl          # Training dataset
│   │   ├── 📜 Scanpy_test.jsonl    # Testing dataset
│   │   ├── 📜 Bowtie2_test.jsonl   # Testing dataset
│   │   ├── 📜 BWA_test.jsonl       # Testing dataset
│   │   ├── 📜 Details.xlsx         # Detailed information about defect functions
│   │   └── ...
│   ├── 📂 Devign/                  # Existing dataset used for comparison
│   └── 📂 REVEAL/                  # Existing dataset used for comparison
│
│── 📁 Classification/            # Implementations of classification models
│   ├── 🤖 Test1_bert/              # BERT model implementation
│   │   ├── 📜 clss_indices.json    # Label mapping file
│   │   ├── 📜 model.py             # Model definition
│   │   ├── 📜 run.py               # Script for fine-tuning the model
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

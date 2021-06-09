# Objective

NLP classification of twitter tweets into one of six emotions: love, joy, fear, anger, surprise, sadness.
The dataset is described in https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp?select=train.txt


# Requirements

python3.4 or above

## Necessary Packages (installation with PyPI)

pip install torch, torchvision

pip install transformers


# Training

Fork the repository (and clone).

Run the _train.py_ scripts with desired arguments in your terminal. For example, to train an ELECTRA-based classifier:

_python ./train.py electra_trained.th xlnet --B=8 --lr=0.00001 --epochs=2_

# Experimental Results

| Model Architecture | Test Accuracy (%) |
| ----------------- | :-----------------: |
ELECTRA (base) encoder + classification head | 93.3 |
BERT (base-uncased) encoder + classification head | 92.4 |
RoBERTta (base) encoder + classification head | 92.6 |

### Training Details

- Initialise encoder with _model_
- Batch Size = 8
- Epochs = 2
- Learning Rate = 1e-5

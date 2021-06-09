# Objective

NLP classification of twitter tweets into one of eight emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, and trust.
The dataset is described in https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp?select=train.txt


# Requirements

python3.4 or above

## Necessary Packages (installation with PyPI)

pip install torch, torchvision

pip install transformers

pip install scandir


# Training

Fork the repository (and clone).

Run the _train.py_ scripts with desired arguments in your terminal. For example, to train an ELECTRA-based classifier:

_python ./train.py electra_trained.th xlnet --B=8 --lr=0.00001 --epochs=2_

# Experimental Results

| Model Architecture | Test Accuracy (%) |
| ----------------- | :-----------------: |
ELECTRA (base) encoder + classification head | - |
BERT (base-uncased) encoder + classification head | - |
RoBERTta (base) encoder + classification head | - |

### Training Details

- Initialise encoder with _model_
- Batch Size = 8
- Epochs = TODO
- Learning Rate = TODO

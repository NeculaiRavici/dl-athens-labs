# Practice in Deep Learning — ATHENS week, November 2026

Labs for the ATHENS course *Practice in Deep Learning* (TP 40, Institut Mines-Télécom), 16–19 November 2026.
All notebooks are written in PyTorch and are meant to be run in Google Colab (GPU runtime recommended).

| Lab | Topic | Notebook | Result |
|---|---|---|---|
| 1 | MNIST classification with a fully connected network (MLP) | [`lab1-mnist-mlp`](lab1-mnist-mlp/Neculai_Ravici-Tataranu_LAB1.ipynb) | 98.3% test accuracy after 20 epochs |
| 2 | MNIST classification with a convolutional network (CNN) | [`lab2-mnist-cnn`](lab2-mnist-cnn/Neculai_Ravici-Tataranu_LAB2.ipynb) | Conv2d + MaxPool + Dropout |
| 3 | Transfer learning: LeNet-5 from scratch vs. pretrained ResNet-18 / EfficientNet-B0 (feature extraction) | [`lab3-transfer-learning`](lab3-transfer-learning/Neculai_Ravici-Tataranu_LAB3.ipynb) | pretrained backbone + new head |
| 4.1 | RNN many-to-one: IMDB review sentiment (mean of embeddings vs. LSTM) | [`lab4-rnn/…Lab4.1`](lab4-rnn/Neculai_Ravici-Tataranu_Lab4.1.ipynb) | 84.5% (embedding mean) / 84.9% (LSTM) test accuracy |
| 4.2 | RNN one-to-many: language model on J. S. Bach's Cello Suites (MIDI), generation by sampling | [`lab4-rnn/…Lab4.2`](lab4-rnn/Neculai_Ravici-Tataranu_Lab4.2.ipynb) | 3-layer LSTM, loss 2.96 → 1.58, 100 generated notes |

## Notes

- **Lab 4.1**: the trained embedding puts *fun, masterpiece, excellent, friendship* closest to *great* (Euclidean distance).
- **Lab 4.2**: `cs6-1pre.mid` contains 4 notes of pitch 79, outside the 79-dimensional one-hot (bins 0–78). They are clipped to the highest bin, so all 36 suites are kept (20,901 training sequences, as in the reference run). Generation uses a temperature softmax, which is equivalent to `softmax(logits / T)`.

## Running

Open a notebook in Colab (*File → Upload notebook*), set *Runtime → Change runtime type → GPU*, then *Run all*. The datasets (MNIST, IMDB, the Bach MIDI files) are downloaded by the notebooks themselves.

# Deep Audio Steganography

## Training and predicting

To train a brand new model:
```bash
./train.py [-e epochs] [-s samples] [-b batch] \
    [-f fft] [-d datadir] [--fixedDataset]
```

To predict from a given model:
```bash
./predict.py [--skip-plot] --model path/to/model.hdf5 \
    --secret path/to/secret.wav \
    --cover path/to/cover.wav
```

To check parsing procedures without involving a trained model:
```bash
./transform.py
```


## Dataset

Download the TIMIT dataset and extract it into the root of the project.
Strip the dataset's first two folder in the following way:

```bash
data
├── PHONCODE.DOC
├── PROMPTS.TXT
├── README.DOC
├── SPKRINFO.TXT
├── SPKRSENT.TXT
├── TEST
├── test_data.csv
├── TESTSET.DOC
├── TIMITDIC.DOC
├── TIMITDIC.TXT
├── TRAIN
└── train_data.csv
```

To cut/extend dataset to a fixed length, `fix_dataset.sh` can be used.
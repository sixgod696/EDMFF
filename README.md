## EDMFF
```
git clone https://github.com/xavysp/EDMFF.git
cd EDMFF
```

### Testing with EDMFF

Copy and paste your images into the `data/` folder, and:

```
python main.py --choose_test_data=-1
```

### Training with EDMFF

Set the following lines in `main.py`:

```
is_testing = False
# training with BIPED
TRAIN_DATA = DATASET_NAMES[0] 
```

Then run:

```
python main.py
```

Check the configurations of the datasets in `dataset.py`.

### UDED dataset

Here is the [link](https://github.com/xavysp/UDED) to access the UDED dataset for edge detection.

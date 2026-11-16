import numpy as np
from examples.logistic_regression.experiment import *

def test_dataset_split_and_baseline():
    xtr, xte, ytr, yte = load_data()
    assert xtr.shape[0] + xte.shape[0] == 569
    acc, _ = baseline_accuracy(xtr, xte, ytr, yte)
    assert acc > 0.9

def test_encrypted_one_step_matches_plaintext_graph():
    x, _, y, _ = load_data()
    w = np.zeros(x.shape[1] + 1)
    assert np.allclose(encrypted_one_step(x, y, w), train_one_step_reference(x, y))

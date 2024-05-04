# Data Science HW3

## Run the code
```python
python train.py --es_iters 30 --epochs 150
```

## Dataset
* Unknown graph data
  * Label Split:
    * Train: 60, Valid: 600, Test: 1200
* File name description
```
  dataset
  │   ├── private_features.pkl # node feature
  │   ├── private_graph.pkl # graph edges
  │   ├── private_num_classes.pkl # number of classes
  │   ├── private_test_labels.pkl # X
  │   ├── private_test_mask.pkl # nodes indices of testing set
  │   ├── private_train_labels.pkl # nodes labels of training set
  │   ├── private_train_mask.pkl # nodes indices of training set
  │   ├── private_val_labels.pkl # nodes labels of validation set
  │   └── private_val_mask.pkl # nodes indices of validation set
```

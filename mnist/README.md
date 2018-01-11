# Basic MNIST Example

```bash
pip install -r requirements.txt
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```
root folder with raw images
```bash
tree ./
  ./
  |-- main.py
  |-- processed
  |-- raw
  |   |-- t10k-images-idx3-ubyte.gz
  |   |-- t10k-labels-idx1-ubyte.gz
  |   |-- train-images-idx3-ubyte.gz
  |   `-- train-labels-idx1-ubyte.gz
```

root folder with processed files
```bash
tree ./
  ./
  |-- main.py
  |-- processed
  |-- raw
  |   |-- t10k-images-idx3-ubyte
  |   |-- t10k-labels-idx1-ubyte
  |   |-- train-images-idx3-ubyte
  |   |-- train-labels-idx1-ubyte
```

root folder after training
```bash
tree ./
  ./
  |-- main.py
  |-- processed
  |   |-- test.pt
  |   `-- training.pt
  |-- raw
  |   |-- t10k-images-idx3-ubyte
  |   |-- t10k-images-idx3-ubyte.gz
  |   |-- t10k-labels-idx1-ubyte
  |   |-- t10k-labels-idx1-ubyte.gz
  |   |-- train-images-idx3-ubyte
  |   |-- train-images-idx3-ubyte.gz
  |   |-- train-labels-idx1-ubyte
  |   `-- train-labels-idx1-ubyte.gz  
```

# mmap-safetensors

memory-efficient safetensors reader using mmap

## Install

```sh
pip install numpy git+https://github.com/kaeru-shigure/mmap-safetensors.git
```

## Usage

```py
from mmap_safetensors import load_safetensors

weight_dict, metadata = load_safetensors("weight.safetensors")
print(type(weight_dict["linear1.weight"])) # <class 'numpy.memmap'>
```

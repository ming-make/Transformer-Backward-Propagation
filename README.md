## Transformer-Backward-Propagation

### Introduction

This is an implementation of backward propagation of Transformer by Pytorch w/Python3.8 based on the paper: *Attention is all you need*.

### Requirements

```
torch==1.13.1
numpy==1.22.3
```

### Project Structure

```
.
|-- Back_Propagation
|   |-- Encoder.py
|   |-- FFN.py
|   |-- LayerNorm.py
|   |-- MultiHead.py
|   |-- __pycache__
|   |   |-- FFN.cpython-38.pyc
|   |   |-- LayerNorm.cpython-38.pyc
|   |   `-- MultiHead.cpython-38.pyc
|   |-- basic_layer.py
|   `-- requirements.txt
|-- LICENSE
|-- README.md
```

You can test the implementation of back propagation by running the commented code at the bottom of each python file.

### Finished

- [x] FFN Layer
- [x] Linear Layer
- [x] Multi-head Attention Layer
- [x] Encoder Layer

#### To Do

- [ ] Embedding Layer
- [ ] Decoder Layer
- [ ] Encoder
- [ ] Decoder
- [ ] Transformer

**Notice**: Without considering dropout

#### References

- [Deep Learning from Scratch(深度学习入门：基于Python的理论与实现)](https://book.douban.com/subject/30270959/)
- [Natural Language Processing(深度学习进阶：自然语言处理)](https://book.douban.com/subject/35225413/)
- [Pytorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Tensorflow Transformer Tutorial](https://tensorflow.google.cn/tutorials/text/transformer)

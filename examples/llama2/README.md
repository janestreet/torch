# Llama 2 Transformer Example

This example implements Llama 2, the transformer model. The code is a port of the
[pytorch example](https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/llama2_model.py).

For training you can use the
[tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
The training text file should be stored in `data/input.txt`.

Compiling and running the example is done via the following command lines:
```bash
dune build examples/llama2/llama2.exe
dune exec examples/llama2/llama2.exe
```

Here is an example of generated data when training on the Shakespeare
dataset after a couple epochs.
```
MENENIUS:
Nay, let them fright for such a gentlewhere; talk unto
thee, for they love with Bianca: to Well-meantled, he would's wrongs grow went with words?
Came he hope of vantage.

ISABELLA:
She's already.

DUKE VINCENTIO:
Your suit's unproUT:
Listerhook of Hereford him, sweet Kate, my lord;
And for Help and well to call my earth are
successoldiers.' Wis : being conceed him;
'Twar, if he says, the noble gracious liquitsing.
```

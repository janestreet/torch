# ocaml-torch
__ocaml-torch__ provides OCaml bindings for the [PyTorch](https://pytorch.org) tensor library.
This brings to OCaml NumPy-like tensor computations with GPU acceleration and tape-based automatic
differentiation.

These bindings use the [PyTorch C++ API](https://pytorch.org/cppdocs/) and are
mostly automatically generated.
The current GitHub tip corresponds to PyTorch **v2.1**.

## Installation with Libtorch

Torch depends on libtorch, so when you install this package, it will try linking to
libtorch depending on your environment variables.
The code for discovering libtorch is in `src/config/discover.ml`.
In order to change how torch binds to libtorch, you must uninstall and reinstall torch.

To install with any of these methods, after configuring your environment, you may either

* `opam install torch`, or
* build from source:
```bash
git clone https://github.com/janestreet/torch.git
cd torch
make all
```

### Option 1: OPAM switch (CPU only)

The [opam](https://opam.ocaml.org/) libtorch package (an optional dependency) can be
installed, and torch will automatically detect it and build with it. However, it might not
suit your needs if you use any of these:
* Windows operating system,
* ARM processors, or
* GPUs.

### Option 2: Conda

If you've installed libtorch via Conda, ensure that you are in the Conda environment with
the `CONDA_PREFIX` set before installing.

### Option 3: System Libraries

If you have libtorch installed as a system library (e.g. RPM), set `LIBTORCH_USE_SYSTEM=1`
before installing.

### Option 4: Custom Libtorch Location
If you have [downloaded libtorch](https://pytorch.org) somewhere, set
`LIBTORCH=/path/to/libtorch/` before installing.

## Examples

### Utop

__ocaml-torch__ can be used in interactive mode via
[utop](https://github.com/ocaml-community/utop) or
[ocaml-jupyter](https://github.com/akabe/ocaml-jupyter).

Here is a sample utop session:

![utop](./images/utop.png)

### Simple Script

To build a simple torch program, create a file `example.ml`:

```ocaml
open Torch

let () =
  let tensor = Tensor.randn [ 4; 2 ] in
  Tensor.print tensor
```

Then create a `dune` file with the following content:

```ocaml
(executables
  (names example)
  (libraries torch))
```

Run `dune exec example.exe` to compile the program and run it!

Alternatively you can first compile the code via `dune build example.exe` then run the executable
`_build/default/example.exe` (note that building the bytecode target `example.bc` may
not work on macos).

### Demos

* [MNIST tutorial](./examples/mnist/README.md).
* [Finetuning a ResNet-18 model](./examples/pretrained/README.md).
* [Generative Adversarial Networks](./examples/gan/README.md).
* [Running some Python model](./examples/jit/README.md).
* [ResNet examples on CIFAR-10](./examples/cifar/README.md).
* [Character-level RNN](./examples/char_rnn/README.md)
* [Neural Style Transfer](./examples/neural_transfer/README.md)
* [Reinforcement Learning](./examples/reinforcement-learning/README.md)

Some more advanced applications from external repos:

* An [OCaml port of mini-dalle](https://github.com/ArulselvanMadhavan/mini_dalle) by Arulselvan Madhavan.
* Natural Language Processing models based on BERT can be found in the
[ocaml-bert repo](https://github.com/LaurentMazare/ocaml-bert).

## Models and Weights

Various pre-trained computer vision models are implemented in the vision library.
The weight files can be downloaded at the following links:


* ResNet-18 [weights](https://github.com/LaurentMazare/tch-rs/releases/download/mw/resnet18.ot).
* ResNet-34 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet34.ot).
* ResNet-50 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet50.ot).
* ResNet-101 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet101.ot).
* ResNet-152 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet152.ot).
* DenseNet-121 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/densenet121.ot).
* DenseNet-161 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/densenet161.ot).
* DenseNet-169 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/densenet169.ot).
* SqueezeNet 1.0 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/squeezenet1_0.ot).
* SqueezeNet 1.1 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/squeezenet1_1.ot).
* VGG-13 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/vgg13.ot).
* VGG-16 [weights](https://github.com/LaurentMazare/tch-rs/releases/download/mw/vgg16.ot).
* AlexNet [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/alexnet.ot).
* Inception-v3 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/inception-v3.ot).
* MobileNet-v2 [weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/mobilenet-v2.ot).
* EfficientNet
  [b0 weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/efficientnet-b0.ot),
  [b1 weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/efficientnet-b1.ot),
  [b2 weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/efficientnet-b2.ot),
  [b3 weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/efficientnet-b3.ot),
  [b4 weights](https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/efficientnet-b4.ot).

Running the pre-trained models on sample images can the easily be done via:
```bash
dune exec examples/pretrained/predict.exe path/to/resnet18.ot images/tiger.jpg
```

## Internals

ocaml-torch uses extensive code generation to produce bindings to thousands of torch C++ functions.
Read [internals.md](./internals.md) for details.

## Acknowledgements

Many thanks to [@LaurentMazare](https://github.com/LaurentMazare) for the [original
work](https://github.com/LaurentMazare/ocaml-torch) of ocaml-torch.

# ocaml-torch internals

ocaml-torch faces several challenges, including:
* binding to thousands of functions
* avoiding any minor memory leaks in these functions
* quickly cleaning up the memory allocations of tensors when OCaml is done with them

In order to solve this, we have 2 steps of code generation. In this diagram, solid arrows
represent the code generation DAG and dashed arrows represent the code dependency DAG:

![code generation DAGs](./images/codegen_graph.png)

At a high level,

* Declarations.yaml contains the function signatures for the whole Torch C++ API.
* Custom binding generation reads all the declarations, and whenever possible, generating
  * glue code for crossing between C/C++ (the generated C/C++ API),
  * glue code for using the (yet to be generated) OCaml `foreign` functions in OCaml (the generated OCaml wrapper),
  * and `ctypes` bindings.
* Stub generation uses the `ctypes` library, reading the bindings and generating C and
  OCaml stubs. These are just glue code to handle C/OCaml FFI. Note that we have some
  manually-written C++ functions and bindings that get generated stubs.
* There are an extremely small number of manually-written stubs (just 1 as of writing)
  that ctypes cannot handle.
* A combination of the generated OCaml wrapper and manually written wrapper provide an
  actually usable OCaml API. These are further built upon in the main library (not
  pictured).

# Memory management

A large part of this complexity is driven by memory management.

## Avoiding memory leaks

It is challenging to write manual FFI stubs without memory leaks or race conditions. We
use `ctypes` to make sure we get this right on the vast majority of functions. Although it
requires a second code generation step, this spares us from reinventing stub generation.

## Cleaning up tensors

We ensure that tensors are freed when OCaml garbage collects them. To do this, each Tensor
is equipped with a custom finalizer. This could be done on either the C++ or OCaml side.
However, the API to inform OCaml of a tensor's true size in memory only exists in C++ (the
custom block API). Without this, OCaml would not know when to garbage collect on CPU and
would OOM easily.


Note that:

* We have not yet informed OCaml of each tensor's true size, but this is coming soon.
* OCaml is unaware of GPU memory usage. GPU users may need to manually free tensors or
  manaully garbage collect.

### Raw tensors and GC tensors

One wrinkle in this setup is that ctypes cannot handle custom blocks. Since we want the
bulk of our stubs to be generated by ctypes, we create a distinction between `raw_tensor`s
and `gc_tensor`s.

|                    | raw tensor | GC tensor   |
|--------------------|------------|-------------|
| has finalizer?     | no         | yes         |
| GC knows its size? | no         | coming soon |
| FFI input for C?   | no         | yes         |
| FFI output from C? | yes        | no          |
| ctypes type        | void ptr   | void ptr    |

The only way to convert from a `raw_tensor` to `gc_tensor` is with the hand-written,
non-ctypes function `with_tensor_gc`. It is used copiously in the generated OCaml wrapper
code to ensure we only surface GC tensors to the user.

The lifecycle of each tensor looks like this:

1. Some wrapper function `let t = Tensor.foo ()` gets invoked, which makes its way into C++.
2. C++ returns a `raw_tensor` that goes through a regular ctypes stub and makes its way
   back to the OCaml `Tensor.foo` call.
3. Still in `Tensor.foo`, `with_tensor_gc` gets invoked. This goes back into C++ and
   copies the pointer (but not the data) of the tensor to a new custom block. It now has
   known off-heap size and a finalizer to free its memory. This gets returned to OCaml
   with the same memory layout ctypes uses but without going through ctypes.
4. Now `let () = Tensor.bar t` gets invoked. This goes through usual ctypes stubs, since
   `t` looks just like a regular `void ptr` to ctypes.
5. Eventually `t` gets garbage collected. OCaml traverses its blocks and runs the
   finalizer on each one, freeing the tensor's data.

The memory of each tensor (raw or GC) looks like this:

```
             block 1              block2
        ------------------      ----------
root -> | ctypes fat ptr |----> | void * |----> tensor
        ------------------      ----------

```

For GC tensors, `block2` is the one with finalizer and off-heap memory.

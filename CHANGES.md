## Release v0.17.0

* Torch 2.2.
* Automatically detected cuda availability for some installation options.
* Released OCaml lock during backward passes, potentially improving async performance.
* Informed OCaml GC of tensor memory allocations, greatly reducing memory usage.
* Various compilation fixes in bin and examples.
* Fixed bug that returned OCaml floats (doubles) in single precision.
* Fixed bug that performed incorrect reductions in some cases.
* Added various new functionality, e.g. optional string arguments for generated functions,
  Scalar.to_int.

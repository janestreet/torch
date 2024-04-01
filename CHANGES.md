* Make torch device module `Hashable` and `Comparable` so it can used as key for
`Hashtbl` and `Map`.

* Automatically detect cuda availability for some installation options

* Refactor code generation and clarified stubs vs. bindings

* Require OCaml runtime locks before raising the exception.

* fixed Npy lib misnaming in `tensor_tool.ml`

* Release OCaml during backward passes, potentially improving async performance.

* Amend optional OPAM libtorch dependency to correct version

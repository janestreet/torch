(** This is just Sys.opaque_identity. We call it in places where we have to keep the
    tensors alive manually. The tensors will live at least long enough to be passed to
    this function. After this call they can be collected.

    Unfortunately Ctypes does not have the lifetime guarantees that you'd expect. CArray
    does not keep things alive that it's pointing at, so the Ocaml GC can easily rip
    things out from under it. See https://github.com/yallop/ocaml-ctypes/issues/476 *)
val keep_values_alive : 'a list -> unit

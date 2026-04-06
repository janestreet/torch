open Torch_refcounted_bindings.Type_defs

(** This type can be used by other libraries to create tensors. They should call
    [CAMLreturn(prepare_ocaml_tensor(tensor));] in C and use [unwrapped_managed_tensor] as
    the type on the OCaml side. Then call [wrap_managed_tensor] to turn it into a usable
    tensor. *)

type unwrapped_managed_tensor = raw_tensor

val wrap_managed_tensor : unwrapped_managed_tensor -> tensor

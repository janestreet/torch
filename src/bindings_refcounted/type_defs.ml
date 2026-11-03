type raw_tensor

(* We know that these tensors are always heap-allocated in the C code. We only use local
   tensors so that the type system can force correct usage of refcounted tensors. So to
   globalize them we don't need to do anything. *)
type tensor : value mod global
type optimizer
type scalar

let[@inline] globalize_tensor tensor = tensor

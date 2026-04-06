open Base

type raw_tensor
type tensor
type optimizer
type scalar

(* The [Obj.magic] is unfortunate but we know that these tensors are always heap-allocated
   in the C code. We only use local tensors so that the type system can force correct
   usage of refcounted tensors. So to globalize them we don't need to do anything. *)
let globalize_tensor = Obj.magic Obj.magic

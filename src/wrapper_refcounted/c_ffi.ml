open Torch_refcounted_bindings.Type_defs

type unwrapped_managed_tensor = raw_tensor

let wrap_managed_tensor (t : raw_tensor) =
  let t : tensor = Obj.magic t in
  (* The tensor is created with refcount 1. We decrement when the scope is cleaned up so
     the tensor is freed then. *)
  Refcounting.add_to_current_scope t
;;

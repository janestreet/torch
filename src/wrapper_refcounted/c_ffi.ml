open Ctypes
open Torch_refcounted_bindings.Type_defs

type unwrapped_managed_tensor

let wrap_managed_tensor (managed : unwrapped_managed_tensor) : gc_tensor =
  let fatptr =
    Ctypes_ptr.Fat.make
      ~managed:(Some (Obj.repr managed))
      ~reftyp:void
      (Obj.magic managed : nativeint)
  in
  let tensor = unsafe_gc_tensor_of_unit_ptr (CPointer fatptr) in
  (* The tensor is created with refcount 1. We decrement when the scope is cleaned up so
     the tensor is freed then. *)
  Refcounting.add_to_current_scope tensor
;;

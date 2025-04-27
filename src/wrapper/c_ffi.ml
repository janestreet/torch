open Ctypes
open Torch_bindings.Type_defs

type unwrapped_managed_tensor

external finalize_managed_tensor
  :  unwrapped_managed_tensor
  -> unit
  = "finalize_managed_tensor"

let wrap_managed_tensor (managed : unwrapped_managed_tensor) : gc_tensor =
  Gc.finalise finalize_managed_tensor managed;
  let fatptr =
    Ctypes_ptr.Fat.make
      ~managed:(Some (Obj.repr managed))
      ~reftyp:void
      (Obj.magic managed : nativeint)
  in
  unsafe_gc_tensor_of_unit_ptr (CPointer fatptr)
;;

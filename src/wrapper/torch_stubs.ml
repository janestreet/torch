open Ctypes
open Torch_bindings.Type_defs
module C = Torch_bindings.C (Torch_stubs_generated)

external finalize_managed_tensor : Obj.t -> unit = "finalize_managed_tensor"
external make_managed_tensor : Ctypes_ptr.voidp -> Obj.t = "make_managed_tensor"

let with_tensor_gc (raw : raw_tensor) : gc_tensor =
  let addr = unsafe_raw_address_of_raw_tensor raw in
  (* When managed is collected, it will reduce the refcount of the corresponding
     Torch tensor. Storing the ~managed reference in the gc_tensor
     fatptr assures that managed will not be collected until the gc_tensor is
     dropped. *)
  let managed = make_managed_tensor addr in
  Gc.finalise finalize_managed_tensor managed;
  let fatptr = Ctypes_ptr.Fat.make ~managed:(Some managed) ~reftyp:void addr in
  unsafe_gc_tensor_of_unit_ptr (CPointer fatptr)
;;

let to_tensor_list (ptr : raw_tensor ptr) =
  let rec loop ptr acc =
    let tensor : raw_tensor = !@ptr in
    if is_none_raw_tensor tensor
    then acc
    else loop (ptr +@ 1) (with_tensor_gc tensor :: acc)
  in
  let result = loop ptr [] in
  C.free (to_voidp ptr);
  List.rev result
;;

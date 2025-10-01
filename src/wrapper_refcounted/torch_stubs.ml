open Ctypes
open Torch_refcounted_bindings.Type_defs
module C = Torch_refcounted_bindings.C (Torch_refcounted_stubs_generated)

module Manual_stubs : sig
  val with_tensor_gc : raw_tensor -> gc_tensor
  val to_tensor_list : raw_tensor ptr -> gc_tensor list
end = struct
  external make_managed_tensor
    :  Ctypes_ptr.voidp
    -> C_ffi.unwrapped_managed_tensor
    = "make_managed_tensor"

  let with_tensor_gc (raw : raw_tensor) : gc_tensor =
    let addr = unsafe_raw_address_of_raw_tensor raw in
    C_ffi.wrap_managed_tensor (make_managed_tensor addr)
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
end

include Manual_stubs

open Ctypes
open Torch_bindings.Type_defs
module C = Torch_bindings.C (Torch_stubs_generated)

external with_tensor_gc : _ Cstubs_internals.fatptr -> Ctypes_ptr.voidp = "with_tensor_gc"

let with_tensor_gc (raw : raw_tensor) : gc_tensor =
  fatptr_of_raw_tensor raw |> with_tensor_gc |> gc_tensor_of_voidp
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

open Base
open Torch_refcounted_bindings.Type_defs

let keep_values_alive vs = ignore (Sys.opaque_identity vs : 'a list)

let globalize_gc_tensor (gct : gc_tensor @ local) =
  match gct with
  (* We know the inner pointer is never on the stack, so it's safe to Obj.magic it to
     global. See c_ffi.ml *)
  | CPointer ptr -> CPointer (Obj.magic Obj.magic ptr) |> unsafe_gc_tensor_of_unit_ptr
;;

let globalize_gc_tensor_list tensors = List.globalize globalize_gc_tensor tensors

let globalize_gc_tensor_opt_list tensors =
  List.globalize (Option.globalize globalize_gc_tensor) tensors
;;

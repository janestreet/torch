open Ctypes

type raw_tensor
type gc_tensor
type ivalue
type module_
type aoti_runner_cuda
type optimizer
type scalar

val raw_tensor : raw_tensor typ
val gc_tensor : gc_tensor typ
val ivalue : ivalue typ
val module_ : module_ typ
val aoti_runner_cuda : aoti_runner_cuda typ
val optimizer : optimizer typ
val scalar : scalar typ
val none_scalar : scalar
val none_gc_tensor : gc_tensor
val is_none_raw_tensor : raw_tensor -> bool
val unsafe_gc_tensor_of_unit_ptr : unit ptr -> gc_tensor
val unsafe_raw_address_of_raw_tensor : raw_tensor -> Ctypes_ptr.voidp

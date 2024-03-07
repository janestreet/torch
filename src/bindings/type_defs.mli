open Ctypes

type raw_tensor
type gc_tensor
type ivalue
type module_
type optimizer
type scalar

val raw_tensor : raw_tensor typ
val gc_tensor : gc_tensor typ
val ivalue : ivalue typ
val module_ : module_ typ
val optimizer : optimizer typ
val scalar : scalar typ
val none_gc_tensor : gc_tensor
val gc_tensor_of_voidp : Ctypes_ptr.voidp -> gc_tensor
val is_none_raw_tensor : raw_tensor -> bool
val fatptr_of_raw_tensor : raw_tensor -> (Obj.t option, unit) Cstubs_internals.fatptr

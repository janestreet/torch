open Ctypes

type raw_tensor = unit ptr
type gc_tensor = unit ptr
type ivalue = unit ptr
type module_ = unit ptr
type optimizer = unit ptr
type scalar = unit ptr

let raw_tensor : raw_tensor typ = ptr void
let gc_tensor : gc_tensor typ = ptr void
let ivalue : ivalue typ = ptr void
let module_ : module_ typ = ptr void
let optimizer : optimizer typ = ptr void
let scalar : scalar typ = ptr void
let none_gc_tensor = null
let is_none_raw_tensor t = is_null t
let unsafe_gc_tensor_of_unit_ptr t = t

let unsafe_raw_address_of_raw_tensor (t : raw_tensor) =
  let (CPointer t) = t in
  Ctypes_ptr.Fat.unsafe_raw_addr t
;;

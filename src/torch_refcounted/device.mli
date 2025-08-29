type t = Torch_refcounted_core.Device.t =
  | Cpu
  | Cuda of int
[@@deriving bin_io, sexp]

include Core.Comparable.S_binable with type t := t
include Core.Hashable.S_binable with type t := t

val of_string : string -> t
val cuda_if_available : unit -> t
val is_cuda : t -> bool
val get_num_threads : unit -> int
val set_num_threads : int -> unit

type t = Torch_core.Device.t =
  | Cpu
  | Cuda of int
[@@deriving bin_io, sexp]

val cuda_if_available : unit -> t
val is_cuda : t -> bool
val get_num_threads : unit -> int
val set_num_threads : int -> unit

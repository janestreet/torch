type t = Torch_core.Device.t =
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

(** Sets num_threads before calling [f], and sets it back to the previous value after [f]
    returns. *)
val with_num_threads : int -> f:(unit -> 'a) -> 'a

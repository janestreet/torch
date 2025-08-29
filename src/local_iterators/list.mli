open Core

(** Wrapper around Core.List that allows iterating with local variables. To be removed
    when Core.List supports this. *)

val iter_local : 'a list -> f:('a -> unit) -> unit
val iter2_local_exn : 'a list -> 'b list -> f:('a -> 'b -> unit) -> unit
val map_local_input : 'a list -> f:('a -> 'b) -> 'b list
val hd_exn : 'a list -> 'a
val init_local : int -> f:(int -> 'a) -> 'a List.t
val unzip_local : ('a * 'b) list -> 'a list * 'b list

open Core

(** Wrapper around Core.List that allows iterating with local variables. To be removed
    when Core.List supports this. *)

val iter_local : 'a list @ local -> f:('a @ local -> unit) -> unit

val iter2_local_exn
  :  'a list @ local
  -> 'b list @ local
  -> f:('a @ local -> 'b @ local -> unit)
  -> unit

val map_local_input : 'a list @ local -> f:('a @ local -> 'b) -> 'b list
val hd_exn : 'a list @ local -> 'a @ local
val init_local : int -> f:(int -> 'a @ local) @ local -> 'a List.t @ local
val unzip_local : ('a * 'b) list @ local -> 'a list * 'b list @ local

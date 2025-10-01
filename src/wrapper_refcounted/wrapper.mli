open Base

val manual_seed : int -> unit
val set_num_threads : int -> unit
val get_num_threads : unit -> int
val record_memory_history : unit -> unit
val save_memory_snapshot_pickled : output_filename:string -> unit

module Scalar : sig
  type _ t

  val int : int -> int t
  val float : float -> float t
  val to_int64 : int t -> int64
  val to_float : float t -> float
end

module Tensor : sig
  type t = Torch_refcounted_bindings.Type_defs.gc_tensor

  include
    Wrapper_generated_refcounted_intf.S with type t := t and type 'a scalar := 'a Scalar.t

  include module type of Refcounting.For_users

  val new_tensor : unit -> t @ local
  val float_vec : ?kind:[ `double | `float | `half ] -> float list -> t @ local
  val int_vec : ?kind:[ `int | `int16 | `int64 | `int8 | `uint8 ] -> int list -> t @ local
  val of_bigarray : (_, _, Bigarray.c_layout) Bigarray.Genarray.t -> t @ local

  (** Both of the below copy functions lay out tensor memory contiguously in the
      bigstring, ignoring the strides of the underlying tensor.

      The copy is performed to/from the window of the bigstring described by the
      [src_pos]/[dst_pos] and [src_len]/[dst_len] arguments, similar to [Bigstring.blit].

      The length of the window described by [src_len]/[dst_len] is in bytes (matching
      other bigstring APIs) and must be precisely the number of bytes required to copy the
      provided tensor. To that extent, the argument is somewhat redundant. However, it
      exists to make sure mistakes are not silently ignored. *)

  val copy_to_bigstring
    :  src:t @ local
    -> dst:(char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
    -> dst_pos:int
    -> dst_len:int
    -> unit

  val copy_from_bigstring
    :  src:(char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
    -> src_pos:int
    -> src_len:int
    -> dst:t @ local
    -> unit

  val copy_to_bigarray
    :  t @ local
    -> (_, _, Bigarray.c_layout) Bigarray.Genarray.t
    -> unit

  val shape : t @ local -> int list
  val size : t @ local -> int list
  val ndim : t @ local -> int
  val shape1_exn : t @ local -> int
  val shape2_exn : t @ local -> int * int
  val shape3_exn : t @ local -> int * int * int
  val shape4_exn : t @ local -> int * int * int * int
  val kind : t @ local -> Torch_wrapper_types.Kind.packed
  val print_rc_scopes_tensors_and_refcounts : unit -> unit
  val requires_grad : t @ local -> bool
  val grad_set_enabled : bool -> bool

  (* returns the previous state. *)
  val get : t @ local -> int -> t @ local
  val select : t @ local -> dim:int -> index:int -> t @ local
  val float_value : t @ local -> float
  val int_value : t @ local -> int
  val float_get : t @ local -> int list -> float
  val int_get : t @ local -> int list -> int
  val float_set : t @ local -> int list -> float -> unit
  val int_set : t @ local -> int list -> int -> unit
  val fill_float : t @ local -> float -> unit
  val fill_int : t @ local -> int -> unit
  val backward : ?keep_graph:bool -> ?create_graph:bool -> t @ local -> unit

  (* Computes and returns the sum of gradients of outputs w.r.t. the inputs.
     If [create_graph] is set to true, graph of the derivative will be constructed,
     allowing to compute higher order derivative products.
  *)
  val run_backward
    :  ?keep_graph:bool
    -> ?create_graph:bool
    -> t list @ local
    -> t list @ local
    -> t list @ local

  val print : t @ local -> unit
  val to_string : t @ local -> line_size:int -> string
  val sum : t @ local -> t @ local
  val mean : t @ local -> t @ local
  val argmax : ?dim:int -> ?keepdim:bool -> t @ local -> t @ local
  val defined : t @ local -> bool
  val device : t @ local -> Torch_wrapper_types.Device.t

  (* Note: copy_ is effectively memcpy, so that src and dst must have same the shape,
     whereas set_data will point the dst data to wherever src.data is. *)
  val copy_ : t @ local -> src:t @ local -> unit

  (** copy_nonblocking_ requires that the host-side tensor is already pinned. *)
  val copy_nonblocking_ : t @ local -> src:t @ local -> unit

  val set_data : t @ local -> src:t @ local -> unit
  val max : t @ local -> t @ local -> t @ local
  val min : t @ local -> t @ local -> t @ local
  val use_count : t @ local -> int

  module For_testing : sig
    include module type of Refcounting.For_testing
  end
end

module Optimizer : sig
  type t

  val adam
    :  learning_rate:float
    -> beta1:float
    -> beta2:float
    -> weight_decay:float
    -> eps:float
    -> t

  val rmsprop
    :  learning_rate:float
    -> alpha:float
    -> eps:float
    -> weight_decay:float
    -> momentum:float
    -> centered:bool
    -> t

  val sgd
    :  learning_rate:float
    -> momentum:float
    -> dampening:float
    -> weight_decay:float
    -> nesterov:bool
    -> t

  val set_learning_rate : t -> float -> unit
  val set_momentum : t -> float -> unit
  val add_parameters : t -> Tensor.t list @ local -> unit
  val zero_grad : t -> unit
  val step : t -> unit
end

module Serialize : sig
  val save : Tensor.t @ local -> filename:string -> unit
  val load : filename:string -> Tensor.t @ local

  val save_multi
    :  named_tensors:(string * Tensor.t) list @ local
    -> filename:string
    -> unit

  val load_multi : names:string list -> filename:string -> Tensor.t list @ local

  val load_multi_
    :  named_tensors:(string * Tensor.t) list @ local
    -> filename:string
    -> unit

  val load_all : filename:string -> (string * Tensor.t) list @ local
end

module Cuda : sig
  val device_count : unit -> int
  val is_available : unit -> bool
  val cudnn_is_available : unit -> bool
  val set_benchmark_cudnn : bool -> unit
end

module Aoti_runner_cuda : sig
  type t

  (** Load an AOT inductor-compiled model from a shared object [file].

      @param so_path the shared object file containing the AOT compiled model
      @param cubin_dir
        load the cubin files from this directory instead of the directory hardcoded in
        [file]
      @param device load the model onto this device
      @param max_concurrent_executions
        maximum number of concurrent invocations of the model that are possible
        (default: 1) *)
  val load
    :  ?max_concurrent_executions:int
    -> device:Torch_wrapper_types.Device.t
    -> cubin_dir:string
    -> so_path:string
    -> unit
    -> t

  val run_unit : t -> Tensor.t list @ local -> unit
end

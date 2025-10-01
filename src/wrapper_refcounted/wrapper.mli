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

  val new_tensor : unit -> t
  val float_vec : ?kind:[ `double | `float | `half ] -> float list -> t
  val int_vec : ?kind:[ `int | `int16 | `int64 | `int8 | `uint8 ] -> int list -> t
  val of_bigarray : (_, _, Bigarray.c_layout) Bigarray.Genarray.t -> t

  (** Both of the below copy functions lay out tensor memory contiguously in the
      bigstring, ignoring the strides of the underlying tensor.

      The copy is performed to/from the window of the bigstring described by the
      [src_pos]/[dst_pos] and [src_len]/[dst_len] arguments, similar to [Bigstring.blit].

      The length of the window described by [src_len]/[dst_len] is in bytes (matching
      other bigstring APIs) and must be precisely the number of bytes required to copy the
      provided tensor. To that extent, the argument is somewhat redundant. However, it
      exists to make sure mistakes are not silently ignored. *)

  val copy_to_bigstring
    :  src:t
    -> dst:(char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
    -> dst_pos:int
    -> dst_len:int
    -> unit

  val copy_from_bigstring
    :  src:(char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
    -> src_pos:int
    -> src_len:int
    -> dst:t
    -> unit

  val copy_to_bigarray : t -> (_, _, Bigarray.c_layout) Bigarray.Genarray.t -> unit
  val shape : t -> int list
  val size : t -> int list
  val ndim : t -> int
  val shape1_exn : t -> int
  val shape2_exn : t -> int * int
  val shape3_exn : t -> int * int * int
  val shape4_exn : t -> int * int * int * int
  val kind : t -> Torch_wrapper_types.Kind.packed
  val print_rc_scopes_tensors_and_refcounts : unit -> unit
  val requires_grad : t -> bool
  val grad_set_enabled : bool -> bool

  (* returns the previous state. *)
  val get : t -> int -> t
  val select : t -> dim:int -> index:int -> t
  val float_value : t -> float
  val int_value : t -> int
  val float_get : t -> int list -> float
  val int_get : t -> int list -> int
  val float_set : t -> int list -> float -> unit
  val int_set : t -> int list -> int -> unit
  val fill_float : t -> float -> unit
  val fill_int : t -> int -> unit
  val backward : ?keep_graph:bool -> ?create_graph:bool -> t -> unit

  (* Computes and returns the sum of gradients of outputs w.r.t. the inputs.
     If [create_graph] is set to true, graph of the derivative will be constructed,
     allowing to compute higher order derivative products.
  *)
  val run_backward : ?keep_graph:bool -> ?create_graph:bool -> t list -> t list -> t list
  val print : t -> unit
  val to_string : t -> line_size:int -> string
  val sum : t -> t
  val mean : t -> t
  val argmax : ?dim:int -> ?keepdim:bool -> t -> t
  val defined : t -> bool
  val device : t -> Torch_wrapper_types.Device.t

  (* Note: copy_ is effectively memcpy, so that src and dst must have same the shape,
     whereas set_data will point the dst data to wherever src.data is. *)
  val copy_ : t -> src:t -> unit

  (** copy_nonblocking_ requires that the host-side tensor is already pinned. *)
  val copy_nonblocking_ : t -> src:t -> unit

  val set_data : t -> src:t -> unit
  val max : t -> t -> t
  val min : t -> t -> t
  val use_count : t -> int

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
  val add_parameters : t -> Tensor.t list -> unit
  val zero_grad : t -> unit
  val step : t -> unit
end

module Serialize : sig
  val save : Tensor.t -> filename:string -> unit
  val load : filename:string -> Tensor.t
  val save_multi : named_tensors:(string * Tensor.t) list -> filename:string -> unit
  val load_multi : names:string list -> filename:string -> Tensor.t list
  val load_multi_ : named_tensors:(string * Tensor.t) list -> filename:string -> unit
  val load_all : filename:string -> (string * Tensor.t) list
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

  val run_unit : t -> Tensor.t list -> unit
end

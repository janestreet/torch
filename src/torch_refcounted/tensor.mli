open Torch_refcounted_core

(* TODO: proper types for Tensor1D, Tensor2D, Tensor3D, ... ? *)
(* TODO: GADT for array element types ? *)
type t = Torch_refcounted_core.Wrapper.Tensor.t

include module type of Torch_refcounted_core.Wrapper.Tensor with type t := t

(** [set_float2 t i j v] sets the element at index [i] and [j] of bidimensional tensor [t]
    to [v]. *)
val set_float2 : t @ local -> int -> int -> float -> unit

(** [set_float1 t i v] sets the element at index [i] of single dimension tensor [t] to
    [v]. *)
val set_float1 : t @ local -> int -> float -> unit

(** [set_int2 t i j v] sets the element at index [i] and [j] of bidimensional tensor [t]
    to [v]. *)
val set_int2 : t @ local -> int -> int -> int -> unit

(** [set_int1 t i v] sets the element at index [i] of single dimension tensor [t] to [v]. *)
val set_int1 : t @ local -> int -> int -> unit

(** [get_float2 t i j] returns the current value from bidimensional tensor [t] at index
    [i] and [j]. *)
val get_float2 : t @ local -> int -> int -> float

(** [get_float1 t i j] returns the current value from single dimension tensor [t] at index
    [i]. *)
val get_float1 : t @ local -> int -> float

(** [get_int2 t i j] returns the current value from bidimensional tensor [t] at indexex
    [i] and [j]. *)
val get_int2 : t @ local -> int -> int -> int

(** [get_int1 t i j] returns the current value from single dimension tensor [t] at index
    [i]. *)
val get_int1 : t @ local -> int -> int

(** Gets an integer element from an arbitrary dimension tensor. *)
val ( .%{} ) : t @ local -> int list -> int

(** Sets an integer element on an arbitrary dimension tensor. *)
val ( .%{}<- ) : t @ local -> int list -> int -> unit

(** Gets a float element from an arbitrary dimension tensor. *)
val ( .%.{} ) : t @ local -> int list -> float

(** Sets a float element on an arbitrary dimension tensor. *)
val ( .%.{}<- ) : t @ local -> int list -> float -> unit

(** Gets an integer element from a single dimension tensor. *)
val ( .%[] ) : t @ local -> int -> int

(** Sets an integer element on a single dimension tensor. *)
val ( .%[]<- ) : t @ local -> int -> int -> unit

(** Gets a float element from a single dimension tensor. *)
val ( .%.[] ) : t @ local -> int -> float

(** Sets a float element on a single dimension tensor. *)
val ( .%.[]<- ) : t @ local -> int -> float -> unit

(** [no_grad_ t ~f] runs [f] on [t] without tracking gradients for t. *)
val no_grad_ : t @ local -> f:(t @ local -> 'a) @ local -> 'a

val no_grad : (unit -> 'a) -> 'a
val zero_grad : t @ local -> unit

(** Pointwise addition. *)
val ( + ) : t @ local -> t @ local -> t @ local

(** Pointwise substraction. *)
val ( - ) : t @ local -> t @ local -> t @ local

(** Pointwise multiplication. *)
val ( * ) : t @ local -> t @ local -> t @ local

(** Pointwise division. *)
val ( / ) : t @ local -> t @ local -> t @ local

(** [t += u] modifies [t] by adding values from [u] in a pointwise way. *)
val ( += ) : t @ local -> t @ local -> unit

(** [t -= u] modifies [t] by subtracting values from [u] in a pointwise way. *)
val ( -= ) : t @ local -> t @ local -> unit

(** [t *= u] modifies [t] by multiplying values from [u] in a pointwise way. *)
val ( *= ) : t @ local -> t @ local -> unit

(** [t /= u] modifies [t] by dividing values from [u] in a pointwise way. *)
val ( /= ) : t @ local -> t @ local -> unit

(** [~-u] returns the opposite of [t], i.e. the same as [Tensor.(f 0. - t)]. *)
val ( ~- ) : t @ local -> t @ local

(** Pointwise equality. *)
val ( = ) : t @ local -> t @ local -> t @ local

(** [eq t1 t2] returns true if [t1] and [t2] have the same kind, shape, and all their
    elements are identical. *)
val eq : t @ local -> t @ local -> bool

val eq_scalar : t @ local -> _ Scalar.t -> t @ local

(** [mm t1 t2] returns the dot product or matrix multiplication between [t1] and [t2]. *)
val mm : t @ local -> t @ local -> t @ local

(** [f v] returns a scalar tensor with value [v]. *)
val f : float -> t @ local

type create =
  ?requires_grad:bool
  -> ?kind:Kind.packed
  -> ?device:Device.t
  -> ?scale:float
  -> int list
  -> t @ local

(** Creates a tensor with value 0. *)
val zeros : create

(** Creates a tensor with value 1. *)
val ones : create

(** Creates a tensor with random values sampled uniformly between 0 and 1. *)
val rand : create

(** Creates a tensor with random values sampled using a standard normal distribution. *)
val randn : create

(** Creates a tensor from a list of float values. *)
val float_vec
  :  ?kind:[ `double | `float | `half ]
  -> ?device:Device.t
  -> float list
  -> t @ local

(** [to_type t ~type_] returns a tensor similar to [t] but converted to kind [type_]. *)
val to_type : t @ local -> type_:Kind.packed -> t @ local

(** [to_kind t ~kind] returns a tensor similar to [t] but converted to kind [kind]. *)
val to_kind : t @ local -> kind:Kind.packed -> t @ local

(** [kind t] returns the kind of elements hold in tensor [t]. *)
val type_ : t @ local -> Kind.packed

(** [to_device t ~device] returns a tensor identical to [t] but placed on device [device]. *)
val to_device : ?device:Device.t -> t @ local -> t @ local

(** [to_float0 t] returns the value hold in a scalar (0-dimension) tensor. If the
    dimension are incorrect, [None] is returned. *)
val to_float0 : t @ local -> float option

(** [to_float1 t] returns the array of values hold in a single dimension tensor. If the
    dimension are incorrect, [None] is returned. *)
val to_float1 : t @ local -> float array option

(** [to_float2 t] returns the array of values hold in a bidimensional tensor. If the
    dimension are incorrect, [None] is returned. *)
val to_float2 : t @ local -> float array array option

(** [to_float3 t] returns the array of values hold in a tridimensional tensor. If the
    dimension are incorrect, [None] is returned. *)
val to_float3 : t @ local -> float array array array option

(** [to_float0_exn t] returns the value hold in a scalar (0-dimension) tensor. *)
val to_float0_exn : t @ local -> float

(** [to_float1_exn t] returns the array of values hold in a single dimension tensor. *)
val to_float1_exn : t @ local -> float array

(** [to_float2_exn t] returns the array of values hold in a bidimensional tensor. *)
val to_float2_exn : t @ local -> float array array

(** [to_float3_exn t] returns the array of values hold in a tridimensional tensor. *)
val to_float3_exn : t @ local -> float array array array

(** [to_int0 t] returns the value hold in a scalar (0-dimension) tensor. If the dimension
    are incorrect, [None] is returned. *)
val to_int0 : t @ local -> int option

(** [to_int1 t] returns the array of values hold in a single dimension tensor. If the
    dimension are incorrect, [None] is returned. *)
val to_int1 : t @ local -> int array option

(** [to_int2 t] returns the array of values hold in a bidimensional tensor. If the
    dimension are incorrect, [None] is returned. *)
val to_int2 : t @ local -> int array array option

(** [to_int3 t] returns the array of values hold in a tridimensional tensor. If the
    dimension are incorrect, [None] is returned. *)
val to_int3 : t @ local -> int array array array option

(** [to_int0_exn t] returns the value hold in a scalar (0-dimension) tensor. *)
val to_int0_exn : t @ local -> int

(** [to_int1_exn t] returns the array of values hold in a single dimension tensor. *)
val to_int1_exn : t @ local -> int array

(** [to_int2_exn t] returns the array of values hold in a bidimensional tensor. *)
val to_int2_exn : t @ local -> int array array

(** [to_int3_exn t] returns the array of values hold in a tridimensional tensor. *)
val to_int3_exn : t @ local -> int array array array

(** [of_float0 v] creates a scalar (0-dimension) tensor with value v. *)
val of_float0 : ?device:Device.t -> float -> t @ local

(** [of_float1 v] creates a single dimension tensor with values vs. *)
val of_float1 : ?device:Device.t -> float array -> t @ local

(** [of_float2 v] creates a two dimension tensor with values vs. *)
val of_float2 : ?device:Device.t -> float array array -> t @ local

(** [of_float3 v] creates a three dimension tensor with values vs. *)
val of_float3 : ?device:Device.t -> float array array array -> t @ local

(** [of_double0 v] creates a scalar (0-dimension) tensor with value v. *)
val of_double0 : ?device:Device.t -> float -> t @ local

(** [of_double1 v] creates a single dimension tensor with values vs. *)
val of_double1 : ?device:Device.t -> float array -> t @ local

(** [of_double2 v] creates a two dimension tensor with values vs. *)
val of_double2 : ?device:Device.t -> float array array -> t @ local

(** [of_double3 v] creates a three dimension tensor with values vs. *)
val of_double3 : ?device:Device.t -> float array array array -> t @ local

(** [of_int0 v] creates a scalar (0-dimension) tensor with value v. *)
val of_int0 : ?device:Device.t -> int -> t @ local

(** [of_int1 v] creates a single dimension tensor with values vs. *)
val of_int1 : ?device:Device.t -> int array -> t @ local

(** [of_int2 v] creates a two dimension tensor with values vs. *)
val of_int2 : ?device:Device.t -> int array array -> t @ local

(** [of_int3 v] creates a three dimension tensor with values vs. *)
val of_int3 : ?device:Device.t -> int array array array -> t @ local

val conv2d
  :  ?padding:int * int
  -> ?dilation:int * int
  -> ?groups:int
  -> t @ local (* input *)
  -> t @ local (* weight *)
  -> t option @ local (* bias *)
  -> stride:int * int
  -> t @ local

val conv_transpose2d
  :  ?output_padding:int * int
  -> ?padding:int * int
  -> ?dilation:int * int
  -> ?groups:int
  -> t @ local (* input *)
  -> t @ local (* weight *)
  -> t option @ local (* bias *)
  -> stride:int * int
  -> t @ local

val max_pool2d
  :  ?padding:int * int
  -> ?dilation:int * int
  -> ?ceil_mode:bool
  -> ?stride:int * int
  -> t @ local
  -> ksize:int * int
  -> t @ local

val avg_pool2d
  :  ?padding:int * int
  -> ?count_include_pad:bool
  -> ?ceil_mode:bool
  -> ?stride:int * int
  -> ?divisor_override:int
  -> t @ local
  -> ksize:int * int
  -> t @ local

val const_batch_norm : ?momentum:float -> ?eps:float -> t @ local -> t @ local

(** [of_bigarray ba] returns a tensor which shape and kind are based on [ba] and holding
    the same data. *)
val of_bigarray
  :  ?device:Device.t
  -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
  -> t @ local

(** [copy_to_bigarray t ba] copies the data from [t] to [ba]. The dimensions of [ba] and
    its kind of element must match the dimension and kind of [t]. *)
val copy_to_bigarray
  :  t @ local
  -> ('b, 'a, Bigarray.c_layout) Bigarray.Genarray.t
  -> unit

(** [to_bigarray t ~kind] converts [t] to a bigarray using the c layout. [kind] has to be
    compatible with the element kind of [t]. *)
val to_bigarray
  :  t @ local
  -> kind:('a, 'b) Bigarray.kind
  -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t

val cross_entropy_for_logits
  :  ?reduction:Reduction.t
  -> t @ local
  -> targets:t @ local
  -> t @ local

(** [dropout t ~p ~is_training] applies dropout to [t] with probability [p]. If
    [is_training] is [false], [t] is returned. If [is_training] is [true], a tensor
    similar to [t] is returned except that each element has a probability [p] to be
    replaced by [0]. *)
val dropout : t @ local -> p:float (* dropout prob *) -> is_training:bool -> t @ local

val nll_loss : ?reduction:Reduction.t -> t @ local -> targets:t @ local -> t @ local

(** [bce_loss t ~targets] returns the binary cross entropy loss between [t] and [targets].
    Elements of [t] are supposed to represent a probability distribution (according to the
    last dimension of [t]), so should be between 0 and 1 and sum to 1. *)
val bce_loss : ?reduction:Reduction.t -> t @ local -> targets:t @ local -> t @ local

(** [bce_loss_with_logits t ~targets] returns the binary cross entropy loss between [t]
    and [targets]. Elements of [t] are logits, a softmax is used in this function to
    convert them to a probability distribution. *)
val bce_loss_with_logits
  :  ?reduction:Reduction.t
  -> t @ local
  -> targets:t @ local
  -> t @ local

(** [mse_loss t1 t2] returns the square of the difference between [t1] and [t2].
    [reduction] can be used to either keep the whole tensor or reduce it by averaging or
    summing. *)

val mse_loss : ?reduction:Reduction.t -> t @ local -> t @ local -> t @ local
val huber_loss : ?reduction:Reduction.t -> t @ local -> t @ local -> t @ local

(** [pp] is a pretty-printer for tensors to be used in top-levels such as utop or jupyter. *)
val pp : Format.formatter -> t @ local -> unit
[@@ocaml.toplevel_printer]

(** [copy t] returns a new copy of [t] with the same size and data which does not share
    storage with t. *)
val copy : t @ local -> t @ local

(** [shape_str t] returns the shape/size of the current tensor as a string. This is useful
    for pretty printing. *)
val shape_str : t @ local -> string

(** [print_shape ?name t] prints the shape/size of t on stdout. If [name] is provided,
    this is also printed. *)
val print_shape : ?name:string -> t @ local -> unit

(** [minimum t] returns the minimum element of tensor [t]. *)
val minimum : t @ local -> t @ local

(** [maximum t] returns the maximum element of tensor [t]. *)
val maximum : t @ local -> t @ local

(** [flatten t] returns a flattened version of t, flattening the dimensions in
    [start_dim, end_dim] *)
val flatten : ?end_dim:int -> t @ local -> start_dim:int -> t @ local

(** [squeeze_last t] squeezes the last dimension of t, i.e. if this dimension has a size
    of 1 it is removed. *)
val squeeze_last : t @ local -> t @ local

(** [scale t f] returns the result of multiplying tensor t by f. *)
val scale : t @ local -> float -> t @ local

(** [to_list t] returns the list of tensors extracted from the first dimension. This is
    the inverse of [cat ~dim:0]. *)
val to_list : t @ local -> t list @ local

val min_values : t @ local -> dim:int list -> keepdim:bool -> t @ local
val max_values : t @ local -> dim:int list -> keepdim:bool -> t @ local

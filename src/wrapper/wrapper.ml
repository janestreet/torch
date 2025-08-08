open Ctypes
open Torch_stubs
open Wrapper_utils
open Torch_bindings.Type_defs

let ptr_of_string str =
  let len = String.length str in
  let carray = CArray.make Ctypes.char (1 + len) in
  String.iteri (fun i char -> CArray.set carray i char) str;
  CArray.set carray len '\x00';
  CArray.start carray
;;

let ptr_of_strings strings =
  let strings = List.map ptr_of_string strings in
  let start = CArray.(of_list (ptr char) strings |> start) in
  Gc.finalise (fun _ -> ignore (Sys.opaque_identity strings : _ list)) start;
  start
;;

module Tensor = struct
  include Wrapper_generated
  open! C.Tensor

  type t = gc_tensor

  let new_tensor () = new_tensor () |> with_tensor_gc

  let float_vec ?(kind = `float) values =
    let values_len = List.length values in
    let values = CArray.of_list double values |> CArray.start in
    let kind =
      match kind with
      | `float -> Kind.T Float
      | `double -> Kind.T Double
      | `half -> Kind.T Half
    in
    let t = float_vec values values_len (Kind.packed_to_int kind) in
    with_tensor_gc t
  ;;

  let int_vec ?(kind = `int) values =
    let values_len = List.length values in
    let values = List.map Int64.of_int values |> CArray.of_list int64_t |> CArray.start in
    let kind =
      match kind with
      | `uint8 -> Kind.T Uint8
      | `int8 -> Kind.T Int8
      | `int16 -> Kind.T Int16
      | `int -> Kind.T Int
      | `int64 -> Kind.T Int64
    in
    let t = int_vec values values_len (Kind.packed_to_int kind) in
    with_tensor_gc t
  ;;

  let of_bigarray (type a b) (ga : (b, a, Bigarray.c_layout) Bigarray.Genarray.t) =
    let dims = Bigarray.Genarray.dims ga in
    let kind = Bigarray.Genarray.kind ga in
    let tensor_kind =
      match kind with
      | Bigarray.Float32 -> Kind.T Float
      | Bigarray.Float64 -> Kind.T Double
      | Bigarray.Int8_signed -> Kind.T Int8
      | Bigarray.Int8_unsigned -> Kind.T Uint8
      | Bigarray.Char -> Kind.T Uint8
      | Bigarray.Int16_signed -> Kind.T Int16
      | Bigarray.Int32 -> Kind.T Int
      | Bigarray.Int -> Kind.T Int64
      | Bigarray.Int64 -> Kind.T Int64
      | _ -> failwith "unsupported bigarray kind"
    in
    let t =
      tensor_of_data
        (bigarray_start genarray ga |> to_voidp)
        (Array.to_list dims
         |> List.map Int64.of_int
         |> CArray.of_list int64_t
         |> CArray.start)
        (Array.length dims)
        (Bigarray.kind_size_in_bytes kind)
        (Kind.packed_to_int tensor_kind)
    in
    with_tensor_gc t
  ;;

  let copy_to_bigstring
    ~src:t
    ~dst:(b : (char, _, Bigarray.c_layout) Bigarray.Array1.t)
    ~dst_pos
    =
    copy_to_bytes
      t
      (bigarray_start array1 b +@ dst_pos |> to_voidp)
      (Bigarray.Array1.dim b - dst_pos |> Int64.of_int)
  ;;

  let copy_to_bigarray (type a b) t (ga : (b, a, Bigarray.c_layout) Bigarray.Genarray.t) =
    let kind = Bigarray.Genarray.kind ga in
    copy_to_elements
      t
      (bigarray_start genarray ga |> to_voidp)
      (Bigarray.Genarray.dims ga |> Array.fold_left ( * ) 1 |> Int64.of_int)
      (Bigarray.kind_size_in_bytes kind)
  ;;

  let shape t =
    let num_dims = ndim t in
    let carray = CArray.make int num_dims in
    shape t (CArray.start carray);
    CArray.to_list carray
  ;;

  let ndim = ndim
  let size = shape

  let unexpected_shape shape =
    let shape = String.concat ", " (List.map string_of_int shape) in
    Printf.sprintf "unexpected shape <%s>" shape |> failwith
  ;;

  let shape1_exn t =
    match shape t with
    | [ s1 ] -> s1
    | shape -> unexpected_shape shape
  ;;

  let shape2_exn t =
    match shape t with
    | [ s1; s2 ] -> s1, s2
    | shape -> unexpected_shape shape
  ;;

  let shape3_exn t =
    match shape t with
    | [ s1; s2; s3 ] -> s1, s2, s3
    | shape -> unexpected_shape shape
  ;;

  let shape4_exn t =
    match shape t with
    | [ s1; s2; s3; s4 ] -> s1, s2, s3, s4
    | shape -> unexpected_shape shape
  ;;

  let kind t = scalar_type t |> Kind.of_int_exn
  let requires_grad t = if requires_grad t <> 0 then true else false
  let grad_set_enabled b = grad_set_enabled (if b then 1 else 0) <> 0

  let get t index =
    let t = get t index in
    with_tensor_gc t
  ;;

  let float_value t = double_value t (from_voidp int null) 0
  let int_value t = int64_value t (from_voidp int null) 0 |> Int64.to_int

  let float_get t indexes =
    double_value t (CArray.of_list int indexes |> CArray.start) (List.length indexes)
  ;;

  let int_get t indexes =
    int64_value t (CArray.of_list int indexes |> CArray.start) (List.length indexes)
    |> Int64.to_int
  ;;

  let float_set t indexes v =
    double_value_set
      t
      (CArray.of_list int indexes |> CArray.start)
      (List.length indexes)
      v
  ;;

  let int_set t indexes v =
    int64_value_set
      t
      (CArray.of_list int indexes |> CArray.start)
      (List.length indexes)
      (Int64.of_int v)
  ;;

  let fill_float t v = fill_double t v
  let fill_int t i = fill_int64 t (Int64.of_int i)

  let backward ?(keep_graph = false) ?(create_graph = false) t =
    backward t (if keep_graph then 1 else 0) (if create_graph then 1 else 0)
  ;;

  let print = print
  let to_string t ~line_size = to_string t line_size
  let argmax ?dim ?(keepdim = false) t = argmax t ~dim ~keepdim
  let max = maximum
  let min = minimum
  let copy_nonblocking_ t ~src = copy_ t src true
  let copy_ t ~src = copy_ t src false
  let set_data t ~src = set_data t src
  let defined = defined
  let device t = device t |> Device.of_int

  let run_backward ?keep_graph ?(create_graph = false) tensors inputs =
    let keep_graph =
      match keep_graph with
      | None -> create_graph
      | Some keep_graph -> keep_graph
    in
    let out_ = CArray.make raw_tensor (List.length inputs) in
    run_backward
      (CArray.of_list gc_tensor tensors |> CArray.start)
      (List.length tensors)
      (CArray.of_list gc_tensor inputs |> CArray.start)
      (List.length inputs)
      (CArray.start out_)
      (if keep_graph then 1 else 0)
      (if create_graph then 1 else 0);
    keep_values_alive tensors;
    keep_values_alive inputs;
    List.map with_tensor_gc (CArray.to_list out_)
  ;;

  let sum t = sum t ~dtype:(kind t)
  let mean t = mean t ~dtype:(kind t)
  let use_count t = use_count t
end

module Scalar = struct
  module S = C.Scalar
  include (S : module type of S)

  type nonrec _ t = scalar

  let int i =
    let t = int (Int64.of_int i) in
    Gc.finalise free t;
    t
  ;;

  let float f =
    let t = float f in
    Gc.finalise free t;
    t
  ;;
end

module Optimizer = struct
  include C.Optimizer

  type t = optimizer

  let adam ~learning_rate ~beta1 ~beta2 ~weight_decay ~eps =
    let t = adam learning_rate beta1 beta2 weight_decay eps in
    Gc.finalise free t;
    t
  ;;

  let rmsprop ~learning_rate ~alpha ~eps ~weight_decay ~momentum ~centered =
    let centered = if centered then 1 else 0 in
    let t = rmsprop learning_rate alpha eps weight_decay momentum centered in
    Gc.finalise free t;
    t
  ;;

  let sgd ~learning_rate ~momentum ~dampening ~weight_decay ~nesterov =
    let t = sgd learning_rate momentum dampening weight_decay nesterov in
    Gc.finalise free t;
    t
  ;;

  let add_parameters t tensors =
    add_parameters t CArray.(of_list gc_tensor tensors |> start) (List.length tensors);
    keep_values_alive tensors
  ;;
end

module Serialize = struct
  include C.Serialize

  let save t ~filename = save t filename

  let escape s =
    String.map
      (function
        | '.' -> '|'
        | c -> c)
      s
  ;;

  let unescape s =
    String.map
      (function
        | '|' -> '.'
        | c -> c)
      s
  ;;

  let load ~filename = load filename |> with_tensor_gc

  let save_multi ~named_tensors ~filename =
    let names, tensors = List.split named_tensors in
    let names = List.map escape names in
    save_multi
      CArray.(of_list gc_tensor tensors |> start)
      (ptr_of_strings names)
      (List.length named_tensors)
      filename;
    keep_values_alive tensors
  ;;

  let load_multi ~names ~filename =
    let names = List.map escape names in
    let ntensors = List.length names in
    let tensors = CArray.make raw_tensor ntensors in
    load_multi (CArray.start tensors) (ptr_of_strings names) ntensors filename;
    let tensors = List.map with_tensor_gc (CArray.to_list tensors) in
    tensors
  ;;

  let load_multi_ ~named_tensors ~filename =
    let names, tensors = List.split named_tensors in
    let names = List.map escape names in
    load_multi_
      CArray.(of_list gc_tensor tensors |> start)
      (ptr_of_strings names)
      (List.length named_tensors)
      filename;
    keep_values_alive tensors
  ;;

  let load_all ~filename =
    let all_tensors = ref [] in
    let callback =
      coerce
        (Foreign.funptr (string @-> raw_tensor @-> returning void))
        (static_funptr (string @-> raw_tensor @-> returning void))
        (fun tensor_name tensor ->
          all_tensors := (unescape tensor_name, with_tensor_gc tensor) :: !all_tensors)
      [@alert "-deprecated"]
    in
    load_callback filename callback;
    !all_tensors
  ;;
end

module Cuda = struct
  include C.Cuda

  let is_available () = is_available () <> 0
  let cudnn_is_available () = cudnn_is_available () <> 0
  let set_benchmark_cudnn b = set_benchmark_cudnn (if b then 1 else 0)
end

module Ivalue = struct
  module Tag = struct
    type t =
      | None
      | Tensor
      | Double
      | Int
      | Bool
      | Tuple
      | IntList
      | DoubleList
      | BoolList
      | String
      | TensorList
      | GenericList
      | GenericDict
  end

  include C.Ivalue

  type t = ivalue

  let none () =
    let t = none () in
    Gc.finalise free t;
    t
  ;;

  let bool b =
    let t = bool (if b then 1 else 0) in
    Gc.finalise free t;
    t
  ;;

  let tensor tensor_ =
    let t = tensor tensor_ in
    Gc.finalise free t;
    t
  ;;

  let double d =
    let t = double d in
    Gc.finalise free t;
    t
  ;;

  let int64 i =
    let t = int64 i in
    Gc.finalise free t;
    t
  ;;

  let tuple ts =
    let t = tuple CArray.(of_list ivalue ts |> start) (List.length ts) in
    Gc.finalise free t;
    keep_values_alive ts;
    t
  ;;

  let tensor_list ts =
    let t = tensor_list CArray.(of_list gc_tensor ts |> start) (List.length ts) in
    Gc.finalise free t;
    keep_values_alive ts;
    t
  ;;

  let string s =
    let t = string s in
    Gc.finalise free t;
    t
  ;;

  let tag t : Tag.t =
    match tag t with
    | 0 -> None
    | 1 -> Tensor
    | 2 -> Double
    | 3 -> Int
    | 4 -> Bool
    | 5 -> Tuple
    | 6 -> IntList
    | 7 -> DoubleList
    | 8 -> BoolList
    | 9 -> String
    | 10 -> TensorList
    | 12 -> GenericList
    | 13 -> GenericDict
    | otherwise -> Printf.sprintf "unexpected tag %d" otherwise |> failwith
  ;;

  let to_bool t = to_bool t <> 0
  let to_tensor t = to_tensor t |> with_tensor_gc

  let to_tuple t =
    let noutputs = tuple_length t in
    let outputs = CArray.make ivalue noutputs in
    to_tuple t (CArray.start outputs) noutputs;
    let outputs = CArray.to_list outputs in
    List.iter (Gc.finalise free) outputs;
    outputs
  ;;

  let to_tensor_list t =
    let noutputs = list_length t in
    let outputs = CArray.make raw_tensor noutputs in
    C.Ivalue.to_tensor_list t (CArray.start outputs) noutputs;
    CArray.to_list outputs |> List.map with_tensor_gc
  ;;
end

module Module = struct
  include C.Module

  type t = module_

  let forward t tensors =
    let t =
      forward t CArray.(of_list gc_tensor tensors |> start) (List.length tensors)
      |> with_tensor_gc
    in
    keep_values_alive tensors;
    t
  ;;

  let forward_ t ivalues =
    let ivalue =
      forward_ t CArray.(of_list ivalue ivalues |> start) (List.length ivalues)
    in
    Gc.finalise C.Ivalue.free ivalue;
    keep_values_alive ivalues;
    ivalue
  ;;

  let named_buffers t =
    let wrapper_ivalue = named_buffers t in
    let names_and_tensors = CArray.make ivalue 2 in
    C.Ivalue.to_tuple wrapper_ivalue (CArray.start names_and_tensors) 2;
    let names_ivalue = CArray.get names_and_tensors 0
    and tensors_ivalue = CArray.get names_and_tensors 1 in
    let len = C.Ivalue.list_length tensors_ivalue in
    let names = CArray.make ivalue len
    and tensors = CArray.make ivalue len in
    C.Ivalue.to_generic_list names_ivalue (CArray.start names) len;
    C.Ivalue.to_generic_list tensors_ivalue (CArray.start tensors) len;
    let names = names |> CArray.to_list |> List.map C.Ivalue.to_string
    and tensors = tensors |> CArray.to_list |> List.map Ivalue.to_tensor in
    List.combine names tensors |> Base.Map.of_alist_exn (module Base.String)
  ;;

  let load ?device filename =
    let m = load filename (Device.option_to_int device) in
    Gc.finalise free m;
    m
  ;;

  let load_str ?device str =
    let m = load_str str (String.length str) (Device.option_to_int device) in
    Gc.finalise free m;
    m
  ;;
end

module Aoti_runner_cuda = struct
  include C.Aoti_runner_cuda

  type t = aoti_runner_cuda

  let load ?(max_concurrent_executions = 1) ~device ~cubin_dir ~so_path () : t =
    let m = load so_path max_concurrent_executions (Device.to_int device) cubin_dir in
    Gc.finalise free m;
    m
  ;;

  let run_unit t tensors =
    let array = CArray.of_list gc_tensor tensors in
    run_unit t (CArray.start array) (CArray.length array);
    keep_values_alive tensors
  ;;
end

let manual_seed seed = C.manual_seed (Int64.of_int seed)
let set_num_threads = C.set_num_threads
let get_num_threads = C.get_num_threads

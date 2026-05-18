open Core
open C_ffi
open Torch_refcounted_bindings.Type_defs
open Torch_wrapper_types

[%%c {| #include "torch_api.h" |}]

module Tensor = struct
  include Wrapper_generated_refcounted0
  include Wrapper_generated_refcounted1
  include Wrapper_generated_refcounted2
  include Wrapper_generated_refcounted3
  include Wrapper_generated_refcounted4
  include Wrapper_generated_refcounted5
  include Wrapper_generated_refcounted6
  include Wrapper_generated_refcounted7
  include Refcounting.For_users

  type t = tensor [@@deriving globalize]

  let new_tensor () =
    [%c.alloc ({| CAMLreturn(at_new_tensor());|} : raw_tensor value)]
    |> wrap_managed_tensor
  ;;

  let float_vec ?(kind = `float) values =
    let kind =
      match kind with
      | `float -> Kind.T Float
      | `double -> Kind.T Double
      | `half -> Kind.T Half
    in
    let kind = Kind.packed_to_int kind in
    [%c.alloc
      ({|CAMLreturn(at_float_vec(%{values:float list value}, %{kind:int}));|}
       : raw_tensor value)]
    |> wrap_managed_tensor
  ;;

  let int_vec ?(kind = `int) values =
    let kind =
      match kind with
      | `uint8 -> Kind.T Uint8
      | `int8 -> Kind.T Int8
      | `int16 -> Kind.T Int16
      | `int -> Kind.T Int
      | `int64 -> Kind.T Int64
    in
    let kind = Kind.packed_to_int kind in
    [%c.alloc
      ({|CAMLreturn(at_int_vec(%{values:int list value}, %{kind:int}));|}
       : raw_tensor value)]
    |> wrap_managed_tensor
  ;;

  let of_bigarray (type a b) (ga : (b, a, Bigarray.c_layout) Bigarray.Genarray.t) =
    let dims = Bigarray.Genarray.dims ga |> Array.to_list in
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
    let element_size = Bigarray.kind_size_in_bytes kind in
    let kind = Kind.packed_to_int tensor_kind in
    [%c.alloc
      ({|CAMLreturn(at_tensor_of_data(%{ga:(b, a, Bigarray.c_layout) Bigarray.Genarray.t value},
                                      %{dims: int list value}, %{element_size:int},
                                      %{kind:int}));|}
       : raw_tensor value)]
    |> wrap_managed_tensor
  ;;

  let copy_to_bigstring
    ~src:t
    ~dst:(b : (char, _, Bigarray.c_layout) Bigarray.Array1.t)
    ~dst_pos
    ~dst_len
    =
    let t = globalize_tensor t in
    let dst_total_len = Bigarray.Array1.dim b in
    Base.Ordered_collection_common.check_pos_len_exn
      ~pos:dst_pos
      ~len:dst_len
      ~total_length:dst_total_len;
    let dst_pos = Int64.of_int dst_pos in
    let dst_len = Int64.of_int dst_len in
    [%c.alloc
      {|at_copy_to_bytes(%{t:t value},
                         %{b:(char, _, Bigarray.c_layout) Bigarray.Array1.t value},
                         %{dst_pos:Int64.t}, %{dst_len:Int64.t});|}]
  ;;

  let copy_from_bigstring
    ~src:(b : (char, _, Bigarray.c_layout) Bigarray.Array1.t)
    ~src_pos
    ~src_len
    ~dst:t
    =
    let t = globalize_tensor t in
    let src_total_len = Bigarray.Array1.dim b in
    Base.Ordered_collection_common.check_pos_len_exn
      ~pos:src_pos
      ~len:src_len
      ~total_length:src_total_len;
    let src_pos = Int64.of_int src_pos in
    let src_len = Int64.of_int src_len in
    [%c.alloc
      {|at_copy_from_bytes(%{t:t value},
                           %{b:(char, _, Bigarray.c_layout) Bigarray.Array1.t value},
                           %{src_pos:Int64.t}, %{src_len:Int64.t});|}]
  ;;

  let copy_to_bigarray (type a b) t (ga : (b, a, Bigarray.c_layout) Bigarray.Genarray.t) =
    let t = globalize_tensor t in
    let kind = Bigarray.Genarray.kind ga in
    let bigarray_size =
      Bigarray.Genarray.dims ga |> Array.fold ~f:( * ) ~init:1 |> Int64.of_int
    in
    let kind_size = Bigarray.kind_size_in_bytes kind in
    [%c.alloc
      {| at_copy_to_elements(%{t:t value},
                             %{ga:(b, a, Bigarray.c_layout) Bigarray.Genarray.t value},
                             %{bigarray_size:Int64.t}, %{kind_size:int});|}]
  ;;

  let copy_from_bigarray
    (type a b)
    (t : t)
    (ga : (b, a, Bigarray.c_layout) Bigarray.Genarray.t)
    =
    let kind = Bigarray.Genarray.kind ga in
    let t = globalize_tensor t in
    let bigarray_size =
      Bigarray.Genarray.dims ga |> Array.fold ~f:( * ) ~init:1 |> Int64.of_int
    in
    let kind_size = Bigarray.kind_size_in_bytes kind in
    [%c.alloc
      {| at_copy_from_elements(%{t:t value},
                               %{ga:(b, a, Bigarray.c_layout) Bigarray.Genarray.t value},
                               %{bigarray_size:Int64.t}, %{kind_size:int});|}]
  ;;

  let ndim t =
    let t = globalize_tensor t in
    [%c.alloc ({|CAMLreturnT(int, at_dim(%{t:t value}));|} : int)]
  ;;

  let shape t =
    let t = globalize_tensor t in
    [%c.alloc ({|CAMLreturn(at_shape(%{t:t value}));|} : int list value)]
  ;;

  let size = shape

  let unexpected_shape shape =
    let shape = String.concat ~sep:", " (List.map ~f:string_of_int shape) in
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

  let kind t =
    let t = globalize_tensor t in
    [%c.alloc ({|CAMLreturnT(int, at_scalar_type(%{t:t value}));|} : int)]
    |> Kind.of_int_exn
  ;;

  let print_rc_scopes_tensors_and_refcounts () =
    print_rc_scopes_tensors_and_refcounts ~shape ~kind
  ;;

  let requires_grad t =
    let t = globalize_tensor t in
    [%c.alloc ({|CAMLreturnT(int, at_requires_grad(%{t:t value}));|} : int)] <> 0
  ;;

  let grad_set_enabled b =
    let b = Bool.to_int b in
    [%c.alloc ({|CAMLreturnT(int, at_grad_set_enabled(%{b:int}));|} : int)] <> 0
  ;;

  let get t index =
    let t = globalize_tensor t in
    [%c.alloc ({|CAMLreturn(at_get(%{t:t value}, %{index:int}));|} : raw_tensor value)]
    |> wrap_managed_tensor
  ;;

  let float_value t =
    let t = globalize_tensor t in
    [%c.alloc
      ({| CAMLreturnT(double, at_double_value_at_indexes(%{t:t value}, Val_emptylist));|}
       : float)]
  ;;

  let int_value t =
    let t = globalize_tensor t in
    [%c.alloc
      ({| CAMLreturnT(int64_t, at_int64_value_at_indexes(%{t:t value}, Val_emptylist));|}
       : Int64.t)]
    |> Int64.to_int_exn
  ;;

  let float_get t indexes =
    let t = globalize_tensor t in
    [%c.alloc
      ({| CAMLreturnT(double, at_double_value_at_indexes(%{t:t value},
                      %{indexes:int list value}));|}
       : float)]
  ;;

  let int_get t indexes =
    let t = globalize_tensor t in
    [%c.alloc
      ({| CAMLreturnT(int64_t, at_int64_value_at_indexes(%{t:t value},
                      %{indexes:int list value}));|}
       : Int64.t)]
    |> Int64.to_int_exn
  ;;

  let float_set t indexes v =
    let t = globalize_tensor t in
    [%c.alloc
      {|at_set_double_value_at_indexes(%{t:t value}, %{indexes:int list value},
                                       %{v:float});|}]
  ;;

  let int_set t indexes v =
    let t = globalize_tensor t in
    [%c.alloc
      {|at_set_int64_value_at_indexes(%{t:t value}, %{indexes:int list value},
                                      %{v:int});|}]
  ;;

  let fill_float t v =
    let t = globalize_tensor t in
    [%c.alloc {|at_fill_double(%{t:t value}, %{v:float});|}]
  ;;

  let fill_int t i =
    let t = globalize_tensor t in
    [%c.alloc {|at_fill_int64(%{t:t value}, %{i:int});|}]
  ;;

  let backward ?(keep_graph = false) ?(create_graph = false) t =
    let t = globalize_tensor t in
    let keep_graph = Bool.to_int keep_graph in
    let create_graph = Bool.to_int create_graph in
    [%c.alloc {| at_backward(%{t:t value}, %{keep_graph:int}, %{create_graph:int}); |}]
  ;;

  let print t =
    let t = globalize_tensor t in
    [%c.no_alloc {| at_print(%{t:t value}); |}]
  ;;

  let to_string t ~line_size =
    let t = globalize_tensor t in
    [%c.alloc
      ({|CAMLreturn(at_to_string(%{t:t value}, %{line_size:int}));|} : string value)]
  ;;

  let argmax ?dim ?(keepdim = false) t = argmax t ~dim ~keepdim
  let max = maximum
  let min = minimum

  let copy_nonblocking_ t ~src =
    let t = globalize_tensor t in
    let src = globalize_tensor src in
    [%c.alloc {| at_copy_(%{t:t value}, %{src:t value}, 1); |}]
  ;;

  let copy_ t ~src =
    let t = globalize_tensor t in
    let src = globalize_tensor src in
    [%c.alloc {| at_copy_(%{t:t value}, %{src:t value}, 0); |}]
  ;;

  let set_data t ~src =
    let t = globalize_tensor t in
    let src = globalize_tensor src in
    [%c.alloc {| at_set_data(%{t:t value}, %{src:t value});|}]
  ;;

  let defined t =
    let t = globalize_tensor t in
    [%c.alloc ({|CAMLreturnT(int, at_defined(%{t:t value}));|} : int)] <> 0
  ;;

  let device t =
    let t = globalize_tensor t in
    [%c.alloc ({|CAMLreturnT(int, at_device(%{t:t value}));|} : int)] |> Device.of_int
  ;;

  let run_backward ?keep_graph ?(create_graph = false) tensors inputs =
    let tensors = [%globalize: t list] tensors in
    let inputs = [%globalize: t list] inputs in
    let keep_graph =
      match keep_graph with
      | None -> create_graph
      | Some keep_graph -> keep_graph
    in
    let keep_graph = Bool.to_int keep_graph in
    let create_graph = Bool.to_int create_graph in
    [%c.alloc
      ({|CAMLreturn(at_run_backward(%{tensors:t list value}, %{inputs:t list value},
                                    %{keep_graph:int}, %{create_graph:int}));|}
       : raw_tensor list value)]
    |> List.map ~f:wrap_managed_tensor
  ;;

  let sum t = sum t ~dtype:(kind t)
  let mean t = mean t ~dtype:(kind t)

  let use_count t =
    let t = globalize_tensor t in
    [%c.alloc ({|CAMLreturnT(int, at_use_count(%{t:t value}));|} : int)]
  ;;

  module For_testing = struct
    include Refcounting.For_testing
  end
end

module Scalar = struct
  type _ t = scalar

  let to_int64 t =
    [%c.alloc
      ({|CAMLreturn(caml_copy_int64(ats_to_int(%{t:_ t value})));|} : int64 value)]
  ;;

  let to_float t =
    [%c.alloc ({|CAMLreturnT(double, ats_to_float(%{t:_ t value}));|} : float)]
  ;;

  let free t = [%c.alloc {| ats_free(%{t:_ t value}); |}]

  let int i =
    let i = Int64.of_int i in
    let t = [%c.alloc ({|CAMLreturn(ats_int(%{i:Int64.t}));|} : _ t value)] in
    Gc.Expert.add_finalizer_exn t free;
    t
  ;;

  let float f =
    let t = [%c.alloc ({|CAMLreturn(ats_float(%{f:float}));|} : _ t value)] in
    Gc.Expert.add_finalizer_exn t free;
    t
  ;;
end

module Optimizer = struct
  type t = optimizer

  let free t = [%c.alloc {| ato_free(%{t:t value}); |}]

  let adam ~learning_rate ~beta1 ~beta2 ~weight_decay ~eps =
    let t =
      [%c.alloc
        ({|CAMLreturn(ato_adam(%{learning_rate:float}, %{beta1:float}, %{beta2:float},
                               %{weight_decay:float}, %{eps:float}));|}
         : t value)]
    in
    Gc.Expert.add_finalizer_exn t free;
    t
  ;;

  let rmsprop ~learning_rate ~alpha ~eps ~weight_decay ~momentum ~centered =
    let centered = Bool.to_int centered in
    let t =
      [%c.alloc
        ({|CAMLreturn(ato_rmsprop(%{learning_rate:float}, %{alpha:float}, %{eps:float},
                                  %{weight_decay:float}, %{momentum:float},
                                  %{centered:int}));|}
         : t value)]
    in
    Gc.Expert.add_finalizer_exn t free;
    t
  ;;

  let sgd ~learning_rate ~momentum ~dampening ~weight_decay ~nesterov =
    let nesterov = Bool.to_int nesterov in
    let t =
      [%c.alloc
        ({|CAMLreturn(ato_sgd(%{learning_rate:float}, %{momentum:float},
                              %{dampening:float}, %{weight_decay:float},
                              %{nesterov:int}));|}
         : t value)]
    in
    Gc.Expert.add_finalizer_exn t free;
    t
  ;;

  let add_parameters t (tensors @ local) =
    let tensors = [%globalize: Tensor.t list] tensors in
    [%c.alloc {|ato_add_parameters(%{t:t value}, %{tensors:Tensor.t list value});|}]
  ;;

  let set_learning_rate t lr =
    [%c.alloc {|ato_set_learning_rate(%{t:t value}, %{lr:float});|}]
  ;;

  let set_momentum t m = [%c.alloc {|ato_set_momentum(%{t:t value}, %{m:float});|}]
  let zero_grad t = [%c.alloc {|ato_zero_grad(%{t:t value});|}]
  let step t = [%c.alloc {|ato_step(%{t:t value});|}]
end

module Serialize = struct
  let save t ~filename =
    let t = globalize_tensor t in
    [%c.alloc {|at_save(%{t:Tensor.t value}, String_val(%{filename:string value}));|}]
  ;;

  let escape s =
    String.map
      ~f:(function
        | '.' -> '|'
        | c -> c)
      s
  ;;

  let unescape s =
    String.map
      ~f:(function
        | '|' -> '.'
        | c -> c)
      s
  ;;

  let load ~filename =
    [%c.alloc
      ({|CAMLreturn(at_load(String_val(%{filename:string value})));|} : raw_tensor value)]
    |> wrap_managed_tensor
  ;;

  let save_multi ~(named_tensors @ local) ~filename =
    let names, tensors = Torch_local_iterators.List.unzip_local named_tensors in
    let names = [%globalize: string list] names in
    let names = List.map ~f:escape names in
    let tensors = [%globalize: Tensor.t list] tensors in
    [%c.alloc
      {| at_save_multi(%{tensors:Tensor.t list value}, %{names:string list value},
                       String_val(%{filename:string value})); |}]
  ;;

  let load_multi ~names ~filename =
    let names = List.map ~f:escape names in
    [%c.alloc
      ({| CAMLreturn(at_load_multi(%{names:string list value},
                                   String_val(%{filename:string value})));|}
       : raw_tensor list value)]
    |> List.map ~f:wrap_managed_tensor
  ;;

  let load_multi_ ~(named_tensors @ local) ~filename =
    let names, tensors = Torch_local_iterators.List.unzip_local named_tensors in
    let names = [%globalize: string list] names in
    let names = List.map ~f:escape names in
    let tensors = [%globalize: Tensor.t list] tensors in
    [%c.alloc
      {| at_load_multi_(%{tensors:Tensor.t list value}, %{names:string list value},
                        String_val(%{filename:string value}));|}]
  ;;

  let load_all ~filename =
    [%c.alloc
      ({|CAMLreturn(at_load_all(String_val(%{filename:string value})));|}
       : (string * raw_tensor) list value)]
    |> List.map ~f:(fun (name, t) -> unescape name, wrap_managed_tensor t)
  ;;
end

module Cuda = struct
  let device_count () = [%c.alloc ({|CAMLreturnT(int, atc_cuda_device_count());|} : int)]

  let is_available () =
    [%c.alloc ({|CAMLreturnT(int, atc_cuda_is_available());|} : int)] <> 0
  ;;

  let cudnn_is_available () =
    [%c.alloc ({|CAMLreturnT(int, atc_cudnn_is_available());|} : int)] <> 0
  ;;

  let set_benchmark_cudnn b =
    let b = Bool.to_int b in
    [%c.alloc {|atc_set_benchmark_cudnn(%{b:int});|}]
  ;;
end

module Aoti_runner_cuda = struct
  type t

  let free t = [%c.alloc {| aoti_runner_cuda_free(%{t:t value}); |}]

  let load ?(max_concurrent_executions = 1) ~device ~cubin_dir ~so_path () : t =
    let device = Device.to_int device in
    let m =
      [%c.alloc
        ({| CAMLreturn(aoti_runner_cuda_load(String_val(%{so_path:string value}),
                                             %{max_concurrent_executions:int},
                                             %{device:int},
                                             String_val(%{cubin_dir:string value}))); |}
         : t value)]
    in
    Gc.Expert.add_finalizer_exn m free;
    m
  ;;

  let run_unit t tensors =
    let tensors =
      Torch_local_iterators.List.map_local_input ~f:globalize_tensor tensors
    in
    [%c.alloc
      {|aoti_runner_cuda_run_unit(%{t:t value}, %{tensors:Tensor.t list value});|}]
  ;;
end

let manual_seed seed =
  let seed = Int64.of_int seed in
  [%c.no_alloc {|at_manual_seed(%{seed:Int64.t});|}]
;;

let set_num_threads num_threads =
  [%c.no_alloc {|at_set_num_threads(%{num_threads:int});|}]
;;

let get_num_threads () = [%c.no_alloc ({|return at_get_num_threads();|} : int)]
let record_memory_history () = [%c.no_alloc {|torch_record_memory_history();|}]

let save_memory_snapshot_pickled ~output_filename =
  [%c.no_alloc
    {|torch_save_memory_snapshot_pickled(String_val(%{output_filename:string value}));|}]
;;

(* Automatically generate the C++ -> C -> ocaml bindings. This takes as input the
   Descriptions.yaml file that gets generated when building PyTorch from source. *)
open Base
open Stdio

let cpp_filename ~refcounted =
  "torch_" ^ (if refcounted then "refcounted_" else "") ^ "api_generated"
;;

let bindings_filename i = [%string "torch_bindings_generated%{i#Int}.ml"]

let wrapper_filename ~refcounted =
  "wrapper_generated" ^ if refcounted then "_refcounted" else ""
;;

let excluded_functions =
  Set.of_list
    (module String)
    [ "multi_margin_loss"
    ; "multi_margin_loss_out"
    ; "log_softmax_backward_data"
    ; "softmax_backward_data"
    ; "copy_"
    ; "conv_transpose2d_backward_out"
    ; "conv_transpose3d_backward_out"
    ; "slow_conv_transpose2d_backward_out"
    ; "slow_conv_transpose3d_backward_out"
    ; "slow_conv3d_backward_out"
    ; "normal"
    ; "_cufft_set_plan_cache_max_size"
    ; "_cufft_clear_plan_cache"
    ; "backward"
    ; "_backward"
    ; "set_data"
    ; "_amp_non_finite_check_and_unscale_"
    ; "_cummin_helper"
    ; "_cummax_helper"
    ; "retain_grad"
    ; "_validate_sparse_coo_tensor_args"
    ; "_validate_sparse_csr_tensor_args"
    ; "count_nonzero"
    ; "_assert_async"
    ; "gradient"
    ; "linalg_vector_norm"
    ; "linalg_vector_norm_out"
    ; "linalg_matrix_norm"
    ; "linalg_matrix_norm_out"
    ; "histogram"
    ; "histogram_out"
      (* Deactivate normal_out, bernoulli_out as these result in some ambiguous function
         calls. *)
    ; "normal_out"
    ; "bernoulli_out"
    ; "nested_tensor"
    ; "arange_out"
    ; "_to_sparse"
    ; "_to_sparse_out"
    ; "to_sparse_out"
    ; "to_sparse_csr_out"
    ; "to_sparse_csc_out"
    ; "to_sparse_bsr_out"
    ; "to_sparse_bsc_out"
    ; "to_sparse"
    ; "to_sparse_csr"
    ; "to_sparse_csc"
    ; "to_sparse_bsr"
    ; "to_sparse_bsc"
    ; (* Can't build this one due to new combination of optional ScalarType *)
      "_sparse_semi_structured_addmm"
    ]
;;

let no_tensor_options =
  Set.of_list
    (module String)
    [ "zeros_like"
    ; "empty_like"
    ; "full_like"
    ; "ones_like"
    ; "rand_like"
    ; "randint_like"
    ; "randn_like"
    ]
;;

let excluded_prefixes =
  [ "thnn_"; "th_"; "_foreach"; "_amp_foreach"; "linalg_norm"; "_nested_tensor"; "sym_" ]
;;

let excluded_suffixes = [ "_forward"; "_forward_out" ]
let yaml_error yaml ~msg = failwith [%string "%{msg}, %{Yaml.to_string_exn yaml}"]

let extract_bool = function
  | `Bool b -> b
  | `String "true" -> true
  | `String "false" -> false
  | yaml -> yaml_error yaml ~msg:"expected bool"
;;

let extract_list = function
  | `A l -> l
  | yaml -> yaml_error yaml ~msg:"expected list"
;;

let extract_map = function
  | `O map -> Map.of_alist_exn (module String) map
  | yaml -> yaml_error yaml ~msg:"expected map"
;;

let extract_string = function
  | `String s -> s
  (* The yaml spec for torch uses n which is converted to a bool. *)
  | `Bool b -> if b then "y" else "n"
  | `Float f -> Float.to_string f
  | yaml -> yaml_error yaml ~msg:"expected string"
;;

let append_local_mode_if_refcounted
  ?(wrap_input_in_parens = false)
  ?(wrap_output_in_parens = false)
  text
  ~refcounted
  =
  if refcounted
  then (
    let text = if wrap_input_in_parens then "(" ^ text ^ ")" else text in
    let text = text ^ " @ local" in
    if wrap_output_in_parens then "(" ^ text ^ ")" else text)
  else text
;;

module Func = struct
  type arg_type =
    | Bool
    | Int64
    | Int64Option
    | Double
    | DoubleOption
    | Tensor
    | TensorOption (* Tensor.t option *)
    | IntList
    | IntListOption
    | DoubleList
    | TensorList
    | TensorOptions (* Tensor kind and device *)
    | Scalar
    | ScalarType
    | ScalarWithDefault of string
    | Device
    | DeviceOption
    | String
    | StringOption

  type arg =
    { arg_name : string
    ; arg_type : arg_type
    ; is_const : bool
    }

  let ml_arg_type arg ~refcounted =
    match arg.arg_type with
    | Bool -> "bool"
    | Int64 -> if String.( = ) arg.arg_name "reduction" then "Reduction.t" else "int"
    | Int64Option -> "int option"
    | Double -> "float"
    | DoubleOption -> "float option"
    | Tensor -> "t" |> append_local_mode_if_refcounted ~refcounted
    | TensorOption -> "t option" |> append_local_mode_if_refcounted ~refcounted
    | IntList -> "int list"
    | IntListOption -> "int list option"
    | DoubleList -> "float list"
    | TensorList -> "t list" |> append_local_mode_if_refcounted ~refcounted
    | TensorOptions -> "Kind.packed * Device.t"
    | Scalar | ScalarWithDefault _ -> "'a scalar"
    | ScalarType -> "Kind.packed"
    | Device -> "Device.t"
    | DeviceOption -> "Device.t option"
    | String -> "string"
    | StringOption -> "string option"
  ;;

  let is_named_arg arg =
    match arg.arg_name with
    | "self" | "other" | "result" | "input" | "tensor" | "tensors" -> false
    | _ -> true
  ;;

  let is_optional_arg arg =
    match arg.arg_type with
    | ScalarWithDefault _ -> true
    | _ -> false
  ;;

  let move_optional_args_to_front (args : arg list) =
    (* To avoid unneeded unit argument appends *)
    let optional_args, nonoptional_args = List.partition_tf args ~f:is_optional_arg in
    optional_args @ nonoptional_args
  ;;

  type t =
    { name : string
    ; operator_name : string
    ; overload_name : string
    ; args : arg list
    ; returns :
        (* number of tensors that are returned *)
        [ `fixed of int | `dynamic | `bool | `int64_t | `double | `nothing ]
    ; kind : [ `function_ | `method_ ]
    }

  let arg_type_of_string str ~is_nullable =
    match String.lowercase str with
    | "bool" -> Some Bool
    | "int64_t" -> Some (if is_nullable then Int64Option else Int64)
    | "double" -> Some (if is_nullable then DoubleOption else Double)
    | "at::tensor" -> Some (if is_nullable then TensorOption else Tensor)
    | "at::tensoroptions" -> Some TensorOptions
    | "at::intarrayref" -> Some (if is_nullable then IntListOption else IntList)
    | "at::arrayref<double>" -> Some DoubleList
    | "const at::itensorlistref &" | "at::tensorlist" -> Some TensorList
    | "at::device" -> Some (if is_nullable then DeviceOption else Device)
    | "const at::scalar &" | "at::scalar" -> Some Scalar
    | "at::scalartype" -> Some ScalarType
    | "c10::string_view" -> Some (if is_nullable then StringOption else String)
    | _ -> None
  ;;

  let safe_arg_name = function
    | "value" -> "value_"
    | arg_name -> arg_name
  ;;

  let c_typed_args_list t =
    List.map t.args ~f:(fun { arg_name; arg_type; is_const = _ } ->
      let arg_name = safe_arg_name arg_name in
      match arg_type with
      | IntList | IntListOption ->
        Printf.sprintf "int64_t *%s_data, int %s_len" arg_name arg_name
      | DoubleList -> Printf.sprintf "double *%s_data, int %s_len" arg_name arg_name
      | TensorList -> Printf.sprintf "gc_tensor *%s_data, int %s_len" arg_name arg_name
      | TensorOptions -> Printf.sprintf "int %s_kind, int %s_device" arg_name arg_name
      | Int64Option -> Printf.sprintf "int64_t %s_v, int %s_null" arg_name arg_name
      | DoubleOption -> Printf.sprintf "double %s_v, int %s_null" arg_name arg_name
      | StringOption -> Printf.sprintf "char * %s_v, int %s_null" arg_name arg_name
      | otherwise ->
        let simple_type_cstring =
          match otherwise with
          | Bool -> "int"
          | Int64 -> "int64_t"
          | Double -> "double"
          | Tensor | TensorOption -> "gc_tensor"
          | ScalarType -> "int"
          | Device | DeviceOption -> "int"
          | Scalar | ScalarWithDefault _ -> "scalar"
          | String -> "const char *"
          | Int64Option
          | DoubleOption
          | IntList
          | IntListOption
          | DoubleList
          | TensorList
          | TensorOptions
          | StringOption -> assert false
        in
        Printf.sprintf "%s %s" simple_type_cstring arg_name)
    |> String.concat ~sep:", "
  ;;

  let c_typed_args_list_rc t =
    List.map t.args ~f:(fun { arg_name; arg_type; is_const = _ } ->
      let arg_name = safe_arg_name arg_name in
      match arg_type with
      | IntList
      | IntListOption
      | DoubleList
      | TensorList
      | Int64Option
      | DoubleOption
      | StringOption
      | Tensor
      | TensorOption
      | Scalar
      | ScalarWithDefault _ -> `value arg_name
      | TensorOptions -> `string [%string "int %{arg_name}_kind, int %{arg_name}_device"]
      | Bool -> `simple ("int", arg_name)
      | Int64 -> `simple ("int64_t", arg_name)
      | Double -> `simple ("double", arg_name)
      | ScalarType -> `simple ("int", arg_name)
      | Device | DeviceOption -> `simple ("int", arg_name)
      | String -> `simple ("const char *", arg_name))
  ;;

  let c_args_list args =
    List.map args ~f:(fun { arg_name; arg_type; is_const } ->
      let arg_name = safe_arg_name arg_name in
      match arg_type, is_const with
      | Scalar, _ -> "*" ^ arg_name
      | ScalarWithDefault default_value, _ ->
        [%string "%{arg_name} ? *%{arg_name} : c10::Scalar{%{default_value}} "]
      | Tensor, true -> [%string "tensor_from_ocaml(%{arg_name})"]
      | Tensor, false | TensorOption, false -> [%string "%{arg_name}_local"]
      | TensorOption, true ->
        [%string
          "%{arg_name} ? std::make_optional(tensor_from_ocaml(%{arg_name})) : \
           std::nullopt"]
      | Bool, _ -> "(bool)" ^ arg_name
      | IntList, _ -> [%string "torch::IntArrayRef(%{arg_name}_data, %{arg_name}_len)"]
      | IntListOption, _ ->
        [%string
          "%{arg_name}_data == nullptr ? c10::nullopt : \
           c10::optional<torch::IntArrayRef>(torch::IntArrayRef(%{arg_name}_data, \
           %{arg_name}_len))"]
      | DoubleList, _ ->
        [%string "at::ArrayRef<double>(%{arg_name}_data, %{arg_name}_len)"]
      | String, _ -> [%string "std::string(%{arg_name})"]
      | StringOption, _ ->
        [%string
          "%{arg_name}_null ? c10::nullopt : \
           c10::optional<c10::string_view>(%{arg_name}_v)"]
      | TensorList, _ -> [%string "of_carray_tensor(%{arg_name}_data, %{arg_name}_len)"]
      | TensorOptions, _ ->
        [%string
          "at::device(device_of_int(%{arg_name}_device)).dtype(at::ScalarType(%{arg_name}_kind))"]
      | Int64Option, _ ->
        [%string
          "%{arg_name}_null ? c10::nullopt : c10::optional<int64_t>(%{arg_name}_v)"]
      | DoubleOption, _ ->
        [%string "%{arg_name}_null ? c10::nullopt : c10::optional<double>(%{arg_name}_v)"]
      | ScalarType, _ -> [%string "torch::ScalarType(%{arg_name})"]
      | Device, _ -> [%string "device_of_int(%{arg_name})"]
      | DeviceOption, _ -> [%string "optional_device_of_int(%{arg_name})"]
      | Int64, _ | Double, _ -> arg_name)
    |> String.concat ~sep:", "
  ;;

  let c_args_list_ppx args =
    List.map args ~f:(fun { arg_name; arg_type; is_const } ->
      let arg_name = safe_arg_name arg_name in
      match arg_type, is_const with
      | Scalar, _ -> [%string "*scalar_from_ocaml_noalloc(%{arg_name})"]
      | ScalarWithDefault default_value, _ ->
        [%string "scalar_option_from_ocaml(%{arg_name}).value_or(%{default_value}) "]
      | Tensor, true -> [%string "rc_tensor_from_ocaml(%{arg_name})"]
      | Tensor, false | TensorOption, false -> [%string "%{arg_name}_local"]
      | TensorOption, true -> [%string "tensor_option_from_ocaml(%{arg_name})"]
      | Bool, _ -> [%string "%{arg_name} != 0"]
      | IntList, _ -> [%string "vec_of_ocaml_int_list(%{arg_name})"]
      | IntListOption, _ ->
        [%string
          "Is_some(%{arg_name}) ? \
           c10::OptionalArrayRef<int64_t>(vec_of_ocaml_int_list(Some_val(%{arg_name}))) \
           : c10::OptionalArrayRef<int64_t>(std::nullopt)"]
      | DoubleList, _ ->
        [%string "at::ArrayRef<double>(vec_of_ocaml_double_list(%{arg_name}))"]
      | String, _ -> [%string "std::string(%{arg_name})"]
      | StringOption, _ -> [%string "optional_string_from_ocaml(%{arg_name})"]
      | TensorList, _ -> [%string "of_ocaml_tensor_list(%{arg_name})"]
      | TensorOptions, _ ->
        [%string
          "at::device(device_of_int(%{arg_name}_device)).dtype(at::ScalarType(%{arg_name}_kind))"]
      | Int64Option, _ -> [%string "optional_int_from_ocaml(%{arg_name})"]
      | DoubleOption, _ -> [%string "optional_double_from_ocaml(%{arg_name})"]
      | ScalarType, _ -> [%string "torch::ScalarType(%{arg_name})"]
      | Device, _ -> [%string "device_of_int(%{arg_name})"]
      | DeviceOption, _ -> [%string "optional_device_of_int(%{arg_name})"]
      | Int64, _ | Double, _ -> arg_name)
    |> String.concat ~sep:", "
  ;;

  let c_call t =
    match t.kind with
    | `function_ -> [%string "torch::%{t.name}(%{c_args_list t.args})"]
    | `method_ ->
      (match t.args with
       | head :: tail ->
         let obj =
           match head.arg_type with
           | Tensor ->
             if head.is_const
             then [%string "tensor_from_ocaml(%{head.arg_name})."]
             else [%string "%{head.arg_name}_local."]
           | _ -> [%string "%{head.arg_name}->"]
         in
         [%string "%{obj}%{t.name}(%{c_args_list tail})"]
       | [] ->
         failwith [%string "Method calls should have at least one argument %{t.name}"])
  ;;

  let c_call_ppx t =
    match t.kind with
    | `function_ -> [%string "torch::%{t.name}(%{c_args_list_ppx t.args})"]
    | `method_ ->
      (match t.args with
       | head :: tail ->
         let obj =
           match head.arg_type with
           | Tensor ->
             if head.is_const
             then [%string "rc_tensor_from_ocaml(%{head.arg_name})."]
             else [%string "%{head.arg_name}_local."]
           | _ -> [%string "%{head.arg_name}->"]
         in
         [%string "%{obj}%{t.name}(%{c_args_list_ppx tail})"]
       | [] ->
         failwith [%string "Method calls should have at least one argument %{t.name}"])
  ;;

  let reclaim_tensor_statements args ~refcounted =
    List.filter_map args ~f:(fun { arg_name; arg_type; is_const; _ } ->
      match arg_type, is_const with
      | Tensor, false | TensorOption, false ->
        let converter =
          if refcounted then "rc_tensor_from_ocaml" else "tensor_from_ocaml"
        in
        Some [%string "    torch::Tensor %{arg_name}_local = %{converter}(%{arg_name});"]
      | _ -> None)
    |> String.concat ~sep:"\n"
  ;;

  let operator_name t =
    match String.lowercase t.operator_name with
    | "scatter_reduce" ->
      (* scatter_reduce is both an operator name and also obtained from the scatter
         operator when using the reduce overload. *)
      "_scatter_reduce"
    | "scatter_reduce_" -> "_scatter_reduce_"
    | other -> other
  ;;

  let bindings_signature t =
    let args =
      List.concat_map t.args ~f:(fun arg ->
        match arg.arg_type with
        | Bool -> [ "int" ]
        | Int64 -> [ "int64_t" ]
        | Int64Option -> [ "int64_t"; "int" ]
        | Double -> [ "double" ]
        | DoubleOption -> [ "double"; "int" ]
        | Tensor | TensorOption -> [ "gc_tensor" ]
        | TensorOptions -> [ "int"; "int" ]
        | ScalarType -> [ "int" ]
        | Device -> [ "int" ]
        | DeviceOption -> [ "int" ]
        | IntList | IntListOption -> [ "ptr int64_t"; "int" ]
        | DoubleList -> [ "ptr double"; "int" ]
        | TensorList -> [ "ptr gc_tensor"; "int" ]
        | String -> [ "string" ]
        | StringOption -> [ "string"; "int" ]
        | Scalar | ScalarWithDefault _ -> [ "scalar" ])
      |> String.concat ~sep:" @-> "
    in
    let simple_binding args return_type =
      if String.length args > 0
      then [%string "%{args} @-> returning %{return_type}"]
      else [%string "void @-> returning %{return_type}"]
    in
    match t.returns with
    | `fixed 1 -> [%string "%{args} @-> returning raw_tensor"]
    | `fixed _ -> [%string "ptr raw_tensor @-> %{args} @-> returning void"]
    | `dynamic -> [%string "%{args} @-> returning (ptr raw_tensor)"]
    | `nothing -> simple_binding args "void"
    | `bool -> simple_binding args "bool"
    | `int64_t -> simple_binding args "int64_t"
    | `double -> simple_binding args "double"
  ;;

  let replace_map =
    Map.of_alist_exn (module String) [ "end", "end_"; "to", "to_"; "t", "tr" ]
  ;;

  let caml_name name =
    Map.find replace_map name |> Option.value ~default:name |> String.lowercase
  ;;

  let needs_unit_append t =
    List.for_all t.args ~f:(fun arg -> is_named_arg arg || is_optional_arg arg)
    && List.exists t.args ~f:is_optional_arg
  ;;

  let caml_args t ~refcounted =
    let arg_strings =
      List.map (move_optional_args_to_front t.args) ~f:(fun arg ->
        let annotated_name =
          match arg.arg_type with
          | Tensor | TensorOption | TensorList ->
            caml_name arg.arg_name
            |> append_local_mode_if_refcounted ~refcounted ~wrap_output_in_parens:true
          | _ -> caml_name arg.arg_name
        in
        if is_optional_arg arg
        then "?" ^ annotated_name
        else if is_named_arg arg
        then "~" ^ annotated_name
        else caml_name annotated_name)
    in
    let arg_strings =
      if needs_unit_append t then arg_strings @ [ "()" ] else arg_strings
    in
    String.concat arg_strings ~sep:" "
  ;;

  let caml_keepalive_args t =
    let filtered =
      List.filter_map t.args ~f:(fun arg ->
        match arg.arg_type with
        | IntList
        | IntListOption
        | Int64Option
        | DoubleOption
        | StringOption
        | DoubleList
        | Bool
        | ScalarType
        | TensorOptions
        | Device
        | DeviceOption
        | Int64
        | TensorOption
        | Double
        | String
        | Scalar
        | ScalarWithDefault _
        | Tensor -> None
        | TensorList -> Some [%string "keep_values_alive %{caml_name arg.arg_name};"])
    in
    match filtered with
    | [] -> None
    | l -> Some (String.concat l ~sep:" ")
  ;;

  let caml_binding_args t =
    List.map t.args ~f:(fun arg ->
      let name = caml_name arg.arg_name in
      match arg.arg_type with
      | IntList ->
        [%string
          {|(List.map Int64.of_int %{name} |> CArray.of_list int64_t |> CArray.start) (List.length %{name})|}]
      | IntListOption ->
        [%string
          {|(match %{name} with | None -> from_voidp int64_t null | Some v -> List.map Int64.of_int v |> CArray.of_list int64_t |> CArray.start) (match %{name} with | None -> -1 | Some v -> List.length v)|}]
      | Int64Option ->
        (* for this and DoubleOption, the 2nd argument is an indicator for null *)
        [%string
          {| (match %{name} with | None -> Int64.zero | Some v -> Int64.of_int v) (match %{name} with | Some _ -> 0 | None -> 1) |}]
      | DoubleOption ->
        [%string
          {| (Option.value %{name} ~default:0.0) (match %{name} with | Some _ -> 0 | None -> 1) |}]
      | StringOption ->
        [%string
          {| (Option.value %{name} ~default:"") (match %{name} with | Some _ -> 0 | None -> 1) |}]
      | DoubleList ->
        [%string
          {|(%{name} |> CArray.of_list double |> CArray.start) (List.length %{name})|}]
      | TensorList ->
        [%string
          "(CArray.of_list gc_tensor %{name} |> CArray.start) (List.length %{name})"]
      | Bool -> [%string "(if %{name} then 1 else 0)"]
      | ScalarType -> [%string "(Kind.packed_to_int %{name})"]
      | TensorOptions ->
        [%string "(Kind.packed_to_int (fst %{name})) (Device.to_int (snd %{name}))"]
      | Device -> [%string "(Device.to_int %{name})"]
      | DeviceOption -> [%string {| (Device.option_to_int %{name}) |}]
      | Int64 ->
        if String.( = ) name "reduction"
        then "(Reduction.to_int reduction |> Int64.of_int)"
        else [%string "(Int64.of_int %{name})"]
      | TensorOption ->
        [%string "(match %{name} with | Some v -> v | None -> none_gc_tensor)"]
      | ScalarWithDefault _ ->
        [%string "(match %{name} with | Some v -> v | None -> none_scalar)"]
      | Double | String | Scalar | Tensor -> name)
    |> String.concat ~sep:" "
  ;;

  let caml_binding_args_rc t =
    List.filter_map t.args ~f:(fun arg ->
      let name = caml_name arg.arg_name in
      match arg.arg_type with
      (* Nothing to do for these. We pass them to C++ as OCaml values. *)
      | IntList
      | IntListOption
      | Int64Option
      | DoubleOption
      | StringOption
      | DoubleList
      | Double
      | String
      | Scalar
      | ScalarWithDefault _ -> None
      | TensorList ->
        Some
          [%string
            "let %{name} = Torch_local_iterators.List.map_local_input %{name} \
             ~f:globalize_tensor in"]
      | Bool -> None
      | ScalarType -> Some [%string "let %{name} = Kind.packed_to_int %{name} in"]
      | TensorOptions ->
        Some
          [%string
            "let %{name}_kind, %{name}_device = %{name} in\n\
             let %{name}_kind = Kind.packed_to_int %{name}_kind in\n\
             let %{name}_device = Device.to_int %{name}_device in"]
      | Device -> Some [%string "let %{name} = Device.to_int %{name} in"]
      | DeviceOption -> Some [%string "let %{name} = Device.option_to_int %{name} in"]
      | Int64 ->
        if String.( = ) name "reduction"
        then Some [%string "let %{name} = Reduction.to_int %{name} in"]
        else None
      | TensorOption ->
        Some [%string "let %{name} = [%globalize: tensor option] %{name} in"]
      | Tensor -> Some [%string "let %{name} = globalize_tensor %{name} in"])
  ;;

  let caml_binding_args_rc_c t =
    List.map t.args ~f:(fun arg ->
      let name = caml_name arg.arg_name in
      let with_type t = "%{" ^ name ^ " : " ^ t ^ "}" in
      match arg.arg_type with
      | IntList -> with_type "int list value"
      | IntListOption -> with_type "int list option value"
      | Int64Option -> with_type "int option value"
      | DoubleOption -> with_type "float option value"
      | StringOption -> with_type "string option value"
      | DoubleList -> with_type "float list value"
      | TensorList -> with_type "tensor list value"
      | Bool -> "Int_val(" ^ with_type "bool value" ^ ")"
      | ScalarType -> with_type "int"
      | TensorOptions -> "%{" ^ name ^ "_kind : int}, %{" ^ name ^ "_device : int}"
      | Device | DeviceOption -> with_type "int"
      | Int64 -> with_type "int"
      | Tensor -> with_type "tensor value"
      | TensorOption -> with_type "tensor option value"
      | Scalar -> with_type "scalar value"
      | ScalarWithDefault _ -> with_type "scalar option value"
      | Double -> with_type "float"
      | String -> "String_val(" ^ with_type "string value" ^ ")")
    |> String.concat ~sep:", "
  ;;
end

exception Not_a_simple_arg

let read_yaml filename =
  let funcs =
    (* Split the file to avoid Yaml.of_string_exn segfaulting. *)
    In_channel.with_file filename ~f:In_channel.input_lines
    |> List.group ~break:(fun _ l -> String.length l > 0 && Char.( = ) l.[0] '-')
    |> List.concat_map ~f:(fun lines ->
      Yaml.of_string_exn (String.concat lines ~sep:"\n") |> extract_list)
  in
  printf "Read %s, got %d functions.\n%!" filename (List.length funcs);
  List.filter_map funcs ~f:(fun yaml ->
    let map = extract_map yaml in
    let name = Map.find_exn map "name" |> extract_string in
    let operator_name = Map.find_exn map "operator_name" |> extract_string in
    let overload_name = Map.find_exn map "overload_name" |> extract_string in
    let deprecated = Map.find_exn map "deprecated" |> extract_bool in
    let method_of =
      Map.find_exn map "method_of" |> extract_list |> List.map ~f:extract_string
    in
    let arguments = Map.find_exn map "arguments" |> extract_list in
    let returns =
      let is_tensor returns =
        let returns = extract_map returns in
        let return_type = Map.find_exn returns "dynamic_type" |> extract_string in
        String.( = ) return_type "at::Tensor"
      in
      let returns = Map.find_exn map "returns" |> extract_list in
      if List.is_empty returns
      then Some `nothing
      else if List.for_all returns ~f:is_tensor
      then Some (`fixed (List.length returns))
      else (
        match returns with
        | [ returns ] ->
          let return_type =
            Map.find_exn (extract_map returns) "dynamic_type" |> extract_string
          in
          (match return_type with
           | "bool" -> Some `bool
           | "int64_t" -> Some `int64_t
           | "double" -> Some `double
           | "at::TensorList" | "dynamic_type: const c10::List<c10::optional<Tensor>> &"
             -> Some `dynamic
           | _ -> None)
        | [] | _ :: _ :: _ -> None)
    in
    let kind =
      if List.exists method_of ~f:(String.( = ) "namespace")
      then Some `function_
      else if List.exists method_of ~f:(String.( = ) "Tensor")
      then Some `method_
      else None
    in
    if (not deprecated)
       && (not
             (List.exists excluded_prefixes ~f:(fun prefix ->
                String.is_prefix name ~prefix)))
       && (not
             (List.exists excluded_suffixes ~f:(fun suffix ->
                String.is_suffix name ~suffix)))
       && not (Set.mem excluded_functions name)
    then
      Option.both returns kind
      |> Option.bind ~f:(fun (returns, kind) ->
        try
          let args =
            List.filter_map arguments ~f:(fun arg ->
              let arg = extract_map arg in
              let arg_name = Map.find_exn arg "name" |> extract_string in
              let arg_type = Map.find_exn arg "dynamic_type" |> extract_string in
              let is_const =
                Map.find_exn arg "type"
                |> extract_string
                |> String.is_prefix ~prefix:"const "
              in
              let is_nullable =
                Map.find arg "is_nullable"
                |> Option.value_map ~default:false ~f:extract_bool
              in
              let default_value = Map.find arg "default" in
              match Func.arg_type_of_string arg_type ~is_nullable, default_value with
              | Some Scalar, Some default_value when not is_nullable ->
                let default_scalar =
                  default_value |> Yaml.to_string_exn |> String.strip
                in
                Some
                  { Func.arg_name
                  ; arg_type = Func.ScalarWithDefault default_scalar
                  ; is_const
                  }
              | Some TensorOptions, Some _ when Set.mem no_tensor_options name -> None
              | Some arg_type, _ -> Some { Func.arg_name; arg_type; is_const }
              | None, Some _ -> None
              | None, None -> raise Not_a_simple_arg)
          in
          Some { Func.name; operator_name; overload_name; args; returns; kind }
        with
        | Not_a_simple_arg -> None)
    else None)
;;

let p out_channel s =
  Printf.ksprintf
    (fun line ->
      Out_channel.output_string out_channel line;
      Out_channel.output_char out_channel '\n')
    s
;;

let write_func_cpp_ppx ~out_cpp ~out_h ~exported_name ~func =
  let pc fmt = p out_cpp fmt in
  let ph fmt = p out_h fmt in
  let c_typed_args_list = Func.c_typed_args_list_rc func in
  let c_typed_args_str =
    List.map c_typed_args_list ~f:(function
      | `string str -> str
      | `simple (type_, name) -> [%string "%{type_} %{name}"]
      | `value name -> [%string "value %{name}"])
    |> String.concat ~sep:", "
  in
  let reclaim_tensors () =
    let statements = Func.reclaim_tensor_statements func.args ~refcounted:true in
    if not (String.is_empty statements) then pc "%s" statements
  in
  let register_ocaml_values () =
    match
      List.filter_map c_typed_args_list ~f:(function
        | `string _ | `simple _ -> None
        | `value name -> Some name)
      |> List.chunks_of ~length:5
    with
    | [] -> pc "  CAMLparam0();"
    | chunks ->
      List.iteri chunks ~f:(fun i names ->
        let fn_name = if i = 0 then "CAMLparam" else "CAMLxparam" in
        pc "  %s%d(%s);" fn_name (List.length names) (String.concat names ~sep:", "))
  in
  match func.returns with
  | `nothing ->
    ph "void atg_%s(%s);" exported_name c_typed_args_str;
    pc "void atg_%s(%s) {" exported_name c_typed_args_str;
    register_ocaml_values ();
    pc "  PROTECT(";
    reclaim_tensors ();
    pc "    %s;" (Func.c_call_ppx func);
    pc "  )";
    pc "}";
    pc ""
  | (`dynamic | `fixed _ | `bool | `int64_t | `double) as returns ->
    let caml_convert =
      match returns with
      | `dynamic -> "to_ocaml_tensor_list"
      | `fixed 1 -> "rc_tensor_to_ocaml"
      | `fixed _ -> "rc_tensors_to_ocaml_tuple"
      | `bool -> "Val_bool"
      | `int64_t -> "caml_copy_int64"
      | `double -> "caml_copy_double"
    in
    ph "value atg_%s(%s);" exported_name c_typed_args_str;
    pc "value atg_%s(%s) {" exported_name c_typed_args_str;
    register_ocaml_values ();
    pc "  PROTECT(";
    reclaim_tensors ();
    pc "    CAMLreturn(%s(%s));" caml_convert (Func.c_call_ppx func);
    pc "  )";
    pc "}";
    pc ""
;;

let write_func_cpp_ctypes ~out_cpp ~out_h ~exported_name ~func =
  let pc fmt = p out_cpp fmt in
  let ph fmt = p out_h fmt in
  let c_typed_args_list = Func.c_typed_args_list func in
  let reclaim_tensors () =
    let statements = Func.reclaim_tensor_statements func.args ~refcounted:false in
    if not (String.is_empty statements) then pc "%s" statements
  in
  match func.returns with
  | `dynamic ->
    pc "raw_tensor *atg_%s(%s) {" exported_name c_typed_args_list;
    reclaim_tensors ();
    pc "  PROTECT(";
    pc "    auto results__ = %s;" (Func.c_call func);
    (* the returned type is a C++ vector of tensors *)
    pc "    int sz = results__.size();";
    pc "    raw_tensor *out__ = (raw_tensor*)malloc((sz + 1) * sizeof(raw_tensor));";
    pc "    for (int i = 0; i < sz; ++i)";
    pc "      out__[i] = tensor_to_ocaml(results__[i]);";
    pc "    out__[sz] = nullptr;";
    pc "    return out__;";
    pc "  )";
    pc "}";
    pc "";
    ph "raw_tensor *atg_%s(%s);" exported_name c_typed_args_list
  | `nothing ->
    pc "void atg_%s(%s) {" exported_name c_typed_args_list;
    reclaim_tensors ();
    pc "  PROTECT(";
    pc "    %s;" (Func.c_call func);
    pc "  )";
    pc "}";
    pc "";
    ph "void atg_%s(%s);" exported_name c_typed_args_list
  | `fixed 1 ->
    pc "raw_tensor atg_%s(%s) {" exported_name c_typed_args_list;
    reclaim_tensors ();
    pc "  PROTECT(";
    pc "    torch::Tensor results__ = %s;" (Func.c_call func);
    pc "    return tensor_to_ocaml(results__);";
    pc "  )";
    pc "}";
    pc "";
    ph "raw_tensor atg_%s(%s);" exported_name c_typed_args_list
  | `fixed ntensors ->
    pc "void atg_%s(raw_tensor *out__, %s) {" exported_name c_typed_args_list;
    reclaim_tensors ();
    pc "  PROTECT(";
    pc "    auto results__ = %s;" (Func.c_call func);
    for i = 0 to ntensors - 1 do
      pc "    out__[%d] = tensor_to_ocaml(std::get<%d>(results__));" i i
    done;
    pc "  )";
    pc "}";
    pc "";
    ph "void atg_%s(raw_tensor *, %s);" exported_name c_typed_args_list
  | (`bool | `int64_t | `double) as returns ->
    let c_type =
      match returns with
      | `bool -> "int"
      | `int64_t -> "int64_t"
      | `double -> "double"
    in
    pc "%s atg_%s(%s) {" c_type exported_name c_typed_args_list;
    reclaim_tensors ();
    pc "  PROTECT(";
    pc "    return %s;" (Func.c_call func);
    pc "  )";
    pc "}";
    pc "";
    ph "%s atg_%s(%s);" c_type exported_name c_typed_args_list
;;

let write_cpp funcs filename ~refcounted i =
  Out_channel.with_file [%string "%{filename}%{i#Int}.cpp"] ~f:(fun out_cpp ->
    Out_channel.with_file ~append:true (filename ^ ".h") ~f:(fun out_h ->
      let pc fmt = p out_cpp fmt in
      let ph fmt = p out_h fmt in
      pc "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
      if refcounted then pc {|#include "torch_api_for_generated.hpp"|};
      pc "";
      if i = 0
      then (
        ph "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
        ph "");
      Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
        if refcounted
        then write_func_cpp_ppx ~out_cpp ~out_h ~exported_name ~func
        else write_func_cpp_ctypes ~out_cpp ~out_h ~exported_name ~func)))
;;

let write_bindings funcs filename =
  Out_channel.with_file filename ~f:(fun out_channel ->
    let p fmt = p out_channel fmt in
    p "(* THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! *)";
    p "";
    p "open Ctypes";
    p "";
    let funcs = Map.to_alist funcs |> List.chunks_of ~length:100 in
    List.iteri funcs ~f:(fun idx funcs ->
      p "module C%d(F: Cstubs.FOREIGN) = struct" idx;
      p "  open F";
      p "  open Type_defs";
      List.iter funcs ~f:(fun (exported_name, func) ->
        p "  let stubs_%s =" (Func.caml_name exported_name);
        p "    foreign \"atg_%s\"" exported_name;
        p "    (%s)" (Func.bindings_signature func);
        p "");
      p "end");
    p "module C(F: Cstubs.FOREIGN) = struct";
    List.iteri funcs ~f:(fun idx _funcs -> p "  include C%d(F)" idx);
    p "end")
;;

let write_wrapper_impl_ppx ~out_ml ~exported_name ~func =
  let pm fmt = p out_ml fmt in
  let caml_name = Func.caml_name exported_name in
  let arg_string_ml ~indent =
    let indent = String.make indent ' ' in
    Func.caml_binding_args_rc func
    |> List.map ~f:(fun arg -> indent ^ arg)
    |> String.concat ~sep:"\n"
  in
  let arg_string_c = Func.caml_binding_args_rc_c func in
  let arg_inputs =
    if List.is_empty func.args then "()" else Func.caml_args func ~refcounted:true
  in
  let cpp_name = "atg_" ^ exported_name in
  pm "let %s %s =" caml_name arg_inputs;
  (match func.returns with
   | `nothing ->
     pm "%s" (arg_string_ml ~indent:2);
     pm "  %s" [%string "[%c.alloc {|%{cpp_name}(%{arg_string_c});|}]"]
   | (`bool | `int64_t | `double) as result_type ->
     let result_type =
       match result_type with
       | `bool -> "bool"
       | `int64_t -> "int64"
       | `double -> "float"
     in
     pm "%s" (arg_string_ml ~indent:2);
     pm
       "  %s"
       [%string
         "[%c.alloc ({|CAMLreturn(%{cpp_name}(%{arg_string_c}));|} : %{result_type} \
          value)]"]
   | `fixed 1 ->
     pm "%s" (arg_string_ml ~indent:2);
     pm
       "  %s"
       [%string
         "[%c.alloc ({|CAMLreturn(%{cpp_name}(%{arg_string_c}));|} : raw_tensor value)] \
          |> wrap_managed_tensor"]
   | `fixed ntensors ->
     let result_strings = List.init ntensors ~f:(fun i -> [%string "result%{i#Int}"]) in
     let result_concat = String.concat ~sep:", " result_strings in
     let tuple_type =
       List.init ntensors ~f:(fun _ -> "raw_tensor") |> String.concat ~sep:" * "
     in
     pm "%s" (arg_string_ml ~indent:2);
     pm
       "  %s"
       [%string
         "let %{result_concat} = [%c.alloc \
          ({|CAMLreturn(%{cpp_name}(%{arg_string_c}));|} : (%{tuple_type}) value)] in"];
     pm
       "  %s"
       (List.map result_strings ~f:(fun s -> "wrap_managed_tensor " ^ s)
        |> String.concat ~sep:", ")
   | `dynamic ->
     pm "%s" (arg_string_ml ~indent:2);
     pm
       "  %s"
       [%string
         "[%c.alloc ({|CAMLreturn(%{cpp_name}(%{arg_string_c}));|} : raw_tensor list \
          value)] |> List.map ~f:wrap_managed_tensor"]);
  pm ""
;;

let write_wrapper_impl_ctypes ~out_ml ~keep_alive_for_call ~exported_name ~func =
  let pm fmt = p out_ml fmt in
  let caml_name = Func.caml_name exported_name in
  pm "let %s %s =" caml_name (Func.caml_args func ~refcounted:false);
  (match func.returns with
   | `nothing | `bool | `int64_t | `double ->
     keep_alive_for_call
       ~call:[%string "stubs_%{caml_name} %{Func.caml_binding_args func}"]
       (Func.caml_keepalive_args func)
   | `fixed 1 ->
     keep_alive_for_call
       ~call:
         [%string "stubs_%{caml_name} %{Func.caml_binding_args func} |> with_tensor_gc"]
       (Func.caml_keepalive_args func)
   | `fixed ntensors ->
     pm "  let out__ = CArray.make raw_tensor %d in" ntensors;
     pm "  stubs_%s (CArray.start out__) %s;" caml_name (Func.caml_binding_args func);
     for i = 0 to ntensors - 1 do
       pm "  let t%d = CArray.get out__ %d |> with_tensor_gc in" i i
     done;
     Func.caml_keepalive_args func |> Option.iter ~f:(pm "  %s");
     pm "  %s" (List.init ntensors ~f:(Printf.sprintf "t%d") |> String.concat ~sep:", ")
   | `dynamic ->
     keep_alive_for_call
       ~call:
         [%string "stubs_%{caml_name} %{Func.caml_binding_args func} |> to_tensor_list"]
       (Func.caml_keepalive_args func));
  pm ""
;;

let write_wrapper_intf ~out_intf ~exported_name ~(func : Func.t) ~refcounted =
  let pi fmt = p out_intf fmt in
  let caml_name = Func.caml_name exported_name in
  let intf_args =
    List.map (Func.move_optional_args_to_front func.args) ~f:(fun arg ->
      if Func.is_optional_arg arg
      then [%string "?%{Func.caml_name arg.arg_name}:%{Func.ml_arg_type arg ~refcounted}"]
      else if Func.is_named_arg arg
      then [%string "%{Func.caml_name arg.arg_name}:%{Func.ml_arg_type arg ~refcounted }"]
      else Func.ml_arg_type arg ~refcounted)
  in
  let intf_args =
    if Func.needs_unit_append func then intf_args @ [ "unit" ] else intf_args
  in
  let intf_arg_str =
    if List.is_empty intf_args
    then "\n    unit"
    else String.concat ~sep:" ->\n    " intf_args
  in
  pi "  val %s : %s ->" caml_name intf_arg_str;
  let returns =
    match func.returns with
    | `nothing -> "unit"
    | `fixed 1 -> "t" |> append_local_mode_if_refcounted ~refcounted
    | `fixed ntensors ->
      List.init ntensors ~f:(fun _ -> "t")
      |> String.concat ~sep:" * "
      |> append_local_mode_if_refcounted ~refcounted ~wrap_input_in_parens:true
    | `dynamic -> "t list" |> append_local_mode_if_refcounted ~refcounted
    | `bool -> "bool"
    | `int64_t -> "int64"
    | `double -> "float"
  in
  pi "    %s" returns;
  pi ""
;;

let write_wrapper funcs filename ~refcounted i ~is_last =
  Out_channel.with_file [%string "%{filename}%{i#Int}.ml"] ~f:(fun out_ml ->
    Out_channel.with_file ~append:true (filename ^ "_intf.ml") ~f:(fun out_intf ->
      let pm fmt = p out_ml fmt in
      let pi fmt = p out_intf fmt in
      let keep_alive_for_call ~call = function
        | None -> pm "  %s" call
        | Some keep_alive ->
          pm "  let result = %s in" call;
          pm "  %s" keep_alive;
          pm "  result"
      in
      pm "(* THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! *)";
      pm "";
      if refcounted then pm "open! Core";
      let need_ctypes = not refcounted in
      if need_ctypes
      then (
        pm "open Ctypes";
        pm
          "open %s.Type_defs"
          (if refcounted then "Torch_refcounted_bindings" else "Torch_bindings"));
      if refcounted then pm "open C_ffi" else pm "open Torch_stubs";
      pm "open Torch_wrapper_types";
      if refcounted
      then pm "open Torch_refcounted_bindings.Type_defs"
      else pm "open Wrapper_utils";
      if need_ctypes then pm "open C.Generated";
      pm "";
      if refcounted then pm "%s\n" {x|[%%c {| #include "torch_api.h" |}]|x};
      if i = 0
      then (
        pi "(* THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! *)";
        pi "open Torch_wrapper_types";
        pi "";
        pi "module type S = sig";
        pi "  type t";
        pi "  type _ scalar";
        pi "");
      Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
        if refcounted
        then write_wrapper_impl_ppx ~out_ml ~exported_name ~func
        else write_wrapper_impl_ctypes ~out_ml ~keep_alive_for_call ~exported_name ~func;
        write_wrapper_intf ~out_intf ~exported_name ~func ~refcounted);
      if is_last then pi "end"))
;;

let methods =
  let c name args =
    { Func.name
    ; operator_name = name
    ; overload_name = ""
    ; args
    ; returns = `fixed 1
    ; kind = `method_
    }
  in
  let ca arg_name arg_type = { Func.arg_name; arg_type; is_const = true } in
  [ c "grad" [ ca "self" Tensor ]
  ; c "set_requires_grad" [ ca "self" Tensor; ca "r" Bool ]
  ; c "toType" [ ca "self" Tensor; ca "scalar_type" ScalarType ]
  ; c "to" [ ca "self" Tensor; ca "device" Device ]
  ]
;;

let run ~declarations_filename ~gen_bindings ~gen_wrappers ~refcounted ~split =
  let funcs = read_yaml declarations_filename in
  let funcs = methods @ funcs in
  printf "Generating code for %d functions.\n%!" (List.length funcs);
  (* Generate some unique names for overloaded functions. *)
  let funcs =
    List.map funcs ~f:(fun func -> Func.operator_name func, func)
    |> Map.of_alist_multi (module String)
    |> Map.to_alist
    |> List.concat_map ~f:(fun (name, funcs) ->
      match funcs with
      | [] -> assert false
      | [ func ] -> [ name, func ]
      | funcs ->
        let has_empty_overload =
          List.exists funcs ~f:(fun (func : Func.t) -> String.is_empty func.overload_name)
        in
        List.sort funcs ~compare:(fun (f1 : Func.t) (f2 : Func.t) ->
          match Int.compare (String.length f1.name) (String.length f2.name) with
          | 0 -> Int.compare (List.length f1.args) (List.length f2.args)
          | cmp -> cmp)
        |> List.mapi ~f:(fun index (func : Func.t) ->
          let operator_name = Func.operator_name func in
          let overload_name = String.lowercase func.overload_name in
          let name =
            if String.is_empty overload_name || (index = 0 && not has_empty_overload)
            then operator_name
            else if String.is_suffix operator_name ~suffix:"_"
            then operator_name ^ overload_name ^ "_"
            else operator_name ^ "_" ^ overload_name
          in
          name, func))
    |> List.sort ~compare:(Comparable.lift String.compare ~f:fst)
  in
  let funcs =
    let div_ceil a b = (a + b - 1) / b in
    List.chunks_of funcs ~length:(div_ceil (List.length funcs) split)
  in
  List.iteri funcs ~f:(fun i funcs ->
    let funcs = Map.of_alist_exn (module String) funcs in
    if gen_bindings
    then (
      assert (not refcounted);
      write_bindings funcs (bindings_filename i));
    if gen_wrappers
    then (
      write_cpp funcs (cpp_filename ~refcounted) ~refcounted i;
      write_wrapper
        funcs
        (wrapper_filename ~refcounted)
        ~refcounted
        i
        ~is_last:(i = split - 1)))
;;

let command =
  Command.basic
    ~summary:"generate bindings or wrapper code for torch functions"
    (let%map_open.Command declarations_filename =
       flag "declarations" (required string) ~doc:"PATH path to Declarations.yaml"
     and gen_bindings =
       flag "bindings" no_arg ~doc:"if passed in, generate ctypes bindings OCaml code"
     and gen_wrappers =
       flag "wrappers" no_arg ~doc:"if passed in, generate wrapper C++ and OCaml code"
     and refcounted =
       flag
         "refcounted"
         (optional_with_default false bool)
         ~doc:
           "BOOL if set, generated code will use \"@ local\" for refcounted tensors and \
            generated files will be named to indicate they are for the refcounted \
            implementation"
     and split =
       flag
         "split"
         (optional_with_default 1 int)
         ~doc:"INT if set, will split into this many files. Default: 1"
     in
     fun () -> run ~declarations_filename ~gen_bindings ~gen_wrappers ~refcounted ~split)
;;

let () = Command_unix.run command

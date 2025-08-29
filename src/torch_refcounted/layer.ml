open! Base

type t = { apply : Tensor.t -> Tensor.t }
type t_with_training = { apply_with_training : Tensor.t -> is_training:bool -> Tensor.t }

let with_training t =
  let apply_with_training xs ~is_training:_ = t.apply xs in
  { apply_with_training }
;;

type activation =
  | Relu
  | Gelu
  | Softmax
  | Log_softmax
  | Tanh
  | Leaky_relu
  | Sigmoid
  | Hardsigmoid

let dilate_ksize ~ksize ~dilation =
  List.zip_exn ksize dilation |> List.map ~f:(fun (k, d) -> 1 + ((k - 1) / d))
;;

let kaiming_uniform vs ~name ~a ~shape =
  match shape with
  | _output_dim :: input_dim :: dilated_ksize ->
    let fan_in = input_dim * List.fold dilated_ksize ~init:1 ~f:( * ) in
    let std = Float.sqrt (2. /. ((1. +. (a *. a)) *. Float.of_int fan_in)) in
    let bound = Float.sqrt 3. *. std in
    Var_store.new_var vs ~shape ~init:(Uniform (-.bound, bound)) ~name
  | _ -> failwith "invalid kernel size"
;;

let apply ?activation ys =
  match activation with
  | Some Relu -> Tensor.relu ys
  | Some Gelu -> Tensor.gelu ~approximate:"none" ys
  | Some Softmax -> Tensor.softmax ys ~dim:(-1) ~dtype:(T Float)
  | Some Log_softmax -> Tensor.log_softmax ys ~dim:(-1) ~dtype:(T Float)
  | Some Tanh -> Tensor.tanh ys
  | Some Sigmoid -> Tensor.sigmoid ys
  | Some Leaky_relu -> Tensor.leaky_relu ys
  | Some Hardsigmoid -> Tensor.hardsigmoid ys
  | None -> ys
;;

let linear vs ?activation ?(use_bias = true) ?w_init ~input_dim output_dim =
  let w =
    let shape = [ output_dim; input_dim ] in
    match w_init with
    | None -> kaiming_uniform vs ~shape ~a:(Float.sqrt 5.) ~name:"weight"
    | Some init -> Var_store.new_var vs ~shape ~init ~name:"weight"
  in
  let apply =
    if use_bias
    then (
      let bound = 1.0 /. Float.sqrt (Float.of_int input_dim) in
      let b =
        Var_store.new_var
          vs
          ~shape:[ output_dim ]
          ~init:(Uniform (-.bound, bound))
          ~name:"bias"
      in
      fun xs ->
        Tensor.with_rc_scope_tensor (fun () ->
          let y = Tensor.(mm xs (tr w) + b) in
          apply y ?activation [@nontail]))
    else
      fun xs ->
      Tensor.with_rc_scope_tensor (fun () ->
        let y = Tensor.(mm xs (tr w)) in
        apply y ?activation [@nontail])
  in
  { apply }
;;

let conv_weight_bias vs ~output_dim ~input_dim ~groups ~ksize ~dilation ~w_init ~use_bias =
  let dilated_ksize = dilate_ksize ~ksize ~dilation in
  let weight =
    let shape = [ output_dim; input_dim / groups ] @ dilated_ksize in
    match w_init with
    | None -> kaiming_uniform vs ~shape ~a:(Float.sqrt 5.) ~name:"weight"
    | Some init -> Var_store.new_var vs ~shape ~init ~name:"weight"
  in
  let bias =
    if use_bias
    then Some (Var_store.new_var vs ~shape:[ output_dim ] ~init:Zeros ~name:"bias")
    else None
  in
  weight, bias
;;

let conv_transpose_weight_bias
  vs
  ~output_dim
  ~input_dim
  ~groups
  ~ksize
  ~dilation
  ~w_init
  ~use_bias
  =
  let w_init =
    Option.value__local w_init ~default:(Var_store.Init.Normal { mean = 0.; stdev = 0.1 })
  in
  let dilated_ksize = dilate_ksize ~ksize ~dilation in
  let weight =
    Var_store.new_var
      vs
      ~shape:([ input_dim; output_dim / groups ] @ dilated_ksize)
      ~init:w_init
      ~name:"weight"
  in
  let bias =
    if use_bias
    then Some (Var_store.new_var vs ~shape:[ output_dim ] ~init:Zeros ~name:"bias")
    else None
  in
  weight, bias
;;

let conv1d
  vs
  ~ksize
  ~stride
  ?activation
  ?(use_bias = true)
  ?w_init
  ?(padding = 0)
  ?(groups = 1)
  ?(dilation = 1)
  ~input_dim
  output_dim
  =
  let dilation = [ dilation ] in
  let weight, bias =
    conv_weight_bias
      vs
      ~output_dim
      ~input_dim
      ~groups
      ~ksize:[ ksize ]
      ~dilation
      ~w_init
      ~use_bias
  in
  let apply xs =
    Tensor.conv1d
      xs
      ~weight
      ~bias
      ~padding:[ padding ]
      ~stride:[ stride ]
      ~groups
      ~dilation
    |> apply ?activation
  in
  { apply }
;;

let conv_transpose1d
  vs
  ~ksize
  ~stride
  ?activation
  ?(use_bias = true)
  ?w_init
  ?(padding = 0)
  ?(output_padding = 0)
  ?(groups = 1)
  ?(dilation = 1)
  ~input_dim
  output_dim
  =
  let dilation = [ dilation ] in
  let weight, bias =
    conv_transpose_weight_bias
      vs
      ~output_dim
      ~input_dim
      ~groups
      ~ksize:[ ksize ]
      ~dilation
      ~w_init
      ~use_bias
  in
  let apply xs =
    Tensor.conv_transpose1d
      xs
      ~weight
      ~bias
      ~output_padding:[ output_padding ]
      ~padding:[ padding ]
      ~stride:[ stride ]
      ~groups
      ~dilation
    |> apply ?activation
  in
  { apply }
;;

let conv2d
  vs
  ~ksize:(k1, k2)
  ~stride
  ?activation
  ?(use_bias = true)
  ?w_init
  ?(padding = 0, 0)
  ?(groups = 1)
  ?(dilation = 1, 1)
  ~input_dim
  output_dim
  =
  match dilation with
  | d1, d2 ->
    let dilation_list = [ d1; d2 ] in
    let weight, bias =
      conv_weight_bias
        vs
        ~output_dim
        ~input_dim
        ~groups
        ~ksize:[ k1; k2 ]
        ~dilation:dilation_list
        ~w_init
        ~use_bias
    in
    let apply xs =
      Tensor.conv2d ~padding ~dilation ~groups xs weight bias ~stride |> apply ?activation
    in
    { apply }
;;

let conv2d_
  vs
  ~ksize
  ~stride
  ?activation
  ?use_bias
  ?w_init
  ?(padding = 0)
  ?groups
  ?(dilation = 1)
  ~input_dim
  output_dim
  =
  conv2d
    vs
    ~ksize:(ksize, ksize)
    ~stride:(stride, stride)
    ?use_bias
    ?activation
    ?w_init
    ~padding:(padding, padding)
    ?groups
    ~dilation:(dilation, dilation)
    ~input_dim
    output_dim
;;

let conv_transpose2d
  vs
  ~ksize:(k1, k2)
  ~stride
  ?activation
  ?(use_bias = true)
  ?w_init
  ?(padding = 0, 0)
  ?(output_padding = 0, 0)
  ?(groups = 1)
  ?(dilation = 1, 1)
  ~input_dim
  output_dim
  =
  match dilation with
  | d1, d2 ->
    let dilation_list = [ d1; d2 ] in
    let weight, bias =
      conv_transpose_weight_bias
        vs
        ~output_dim
        ~input_dim
        ~groups
        ~ksize:[ k1; k2 ]
        ~dilation:dilation_list
        ~w_init
        ~use_bias
    in
    let apply xs =
      Tensor.conv_transpose2d
        ~output_padding
        ~padding
        ~groups
        ~dilation
        xs
        weight
        bias
        ~stride
      |> apply ?activation
    in
    { apply }
;;

let conv_transpose2d_
  vs
  ~ksize
  ~stride
  ?activation
  ?use_bias
  ?w_init
  ?(padding = 0)
  ?(output_padding = 0)
  ?groups
  ?(dilation = 1)
  ~input_dim
  output_dim
  =
  conv_transpose2d
    vs
    ~ksize:(ksize, ksize)
    ~stride:(stride, stride)
    ?activation
    ?use_bias
    ?w_init
    ~padding:(padding, padding)
    ~output_padding:(output_padding, output_padding)
    ?groups
    ~dilation:(dilation, dilation)
    ~input_dim
    output_dim
;;

let conv3d
  vs
  ~ksize:(k1, k2, k3)
  ~stride:(s1, s2, s3)
  ?activation
  ?(use_bias = true)
  ?w_init
  ?(padding = 0, 0, 0)
  ?(groups = 1)
  ?(dilation = 1, 1, 1)
  ~input_dim
  output_dim
  =
  match padding, dilation with
  | (p1, p2, p3), (d1, d2, d3) ->
    let dilation = [ d1; d2; d3 ] in
    let weight, bias =
      conv_weight_bias
        vs
        ~output_dim
        ~input_dim
        ~groups
        ~ksize:[ k1; k2; k3 ]
        ~dilation
        ~w_init
        ~use_bias
    in
    let apply xs =
      Tensor.conv1d
        xs
        ~weight
        ~bias
        ~padding:[ p1; p2; p3 ]
        ~stride:[ s1; s2; s3 ]
        ~groups
        ~dilation
      |> apply ?activation
    in
    { apply }
;;

let conv3d_
  vs
  ~ksize
  ~stride
  ?activation
  ?use_bias
  ?w_init
  ?(padding = 0)
  ?groups
  ?(dilation = 1)
  ~input_dim
  output_dim
  =
  conv3d
    vs
    ~ksize:(ksize, ksize, ksize)
    ~stride:(stride, stride, stride)
    ?use_bias
    ?activation
    ?w_init
    ~padding:(padding, padding, padding)
    ~dilation:(dilation, dilation, dilation)
    ?groups
    ~input_dim
    output_dim
;;

let conv_transpose3d
  vs
  ~ksize:(k1, k2, k3)
  ~stride:(s1, s2, s3)
  ?activation
  ?(use_bias = true)
  ?w_init
  ?(padding = 0, 0, 0)
  ?(output_padding = 0, 0, 0)
  ?(groups = 1)
  ?(dilation = 1, 1, 1)
  ~input_dim
  output_dim
  =
  match padding, output_padding, dilation with
  | (p1, p2, p3), (o1, o2, o3), (d1, d2, d3) ->
    let dilation = [ d1; d2; d3 ] in
    let weight, bias =
      conv_transpose_weight_bias
        vs
        ~output_dim
        ~input_dim
        ~groups
        ~ksize:[ k1; k2; k3 ]
        ~dilation
        ~w_init
        ~use_bias
    in
    let apply xs =
      Tensor.conv_transpose3d
        ~weight
        ~bias
        ~stride:[ s1; s2; s3 ]
        ~padding:[ p1; p2; p3 ]
        ~output_padding:[ o1; o2; o3 ]
        ~groups
        ~dilation
        xs
      |> apply ?activation
    in
    { apply }
;;

let conv_transpose3d_
  vs
  ~ksize
  ~stride
  ?activation
  ?use_bias
  ?w_init
  ?(padding = 0)
  ?(output_padding = 0)
  ?groups
  ?(dilation = 1)
  ~input_dim
  output_dim
  =
  conv_transpose3d
    vs
    ~ksize:(ksize, ksize, ksize)
    ~stride:(stride, stride, stride)
    ?activation
    ?use_bias
    ?w_init
    ~padding:(padding, padding, padding)
    ~output_padding:(output_padding, output_padding, output_padding)
    ?groups
    ~dilation:(dilation, dilation, dilation)
    ~input_dim
    output_dim
;;

let batch_norm2d
  vs
  ?(w_init = Var_store.Init.Uniform (0., 1.))
  ?(cudnn_enabled = true)
  ?(eps = 1e-5)
  ?(momentum = 0.1)
  output_dim
  =
  let w = Var_store.new_var vs ~shape:[ output_dim ] ~init:w_init ~name:"weight" in
  let b = Var_store.new_var vs ~shape:[ output_dim ] ~init:Zeros ~name:"bias" in
  let running_mean =
    Var_store.new_var
      vs
      ~trainable:false
      ~shape:[ output_dim ]
      ~init:Zeros
      ~name:"running_mean"
  in
  let running_var =
    Var_store.new_var
      vs
      ~trainable:false
      ~shape:[ output_dim ]
      ~init:Ones
      ~name:"running_var"
  in
  let apply_with_training xs ~is_training =
    Tensor.batch_norm
      xs
      ~weight:(Some w)
      ~bias:(Some b)
      ~running_mean:(Some running_mean)
      ~running_var:(Some running_var)
      ~training:is_training
      ~momentum
      ~eps
      ~cudnn_enabled
  in
  { apply_with_training }
;;

let layer_norm vs ?(cudnn_enable = true) ?(eps = 1e-5) dim =
  let weight = Var_store.new_var vs ~name:"weight" ~shape:[ dim ] ~init:Ones in
  let bias = Var_store.new_var vs ~name:"bias" ~shape:[ dim ] ~init:Zeros in
  let apply xs =
    Tensor.layer_norm
      xs
      ~normalized_shape:[ dim ]
      ~weight:(Some weight)
      ~bias:(Some bias)
      ~eps
      ~cudnn_enable
  in
  { apply }
;;

let forward t xs = Tensor.with_rc_scope_tensor (fun () -> t.apply xs)

let forward_ t_with_training xs ~is_training =
  Tensor.with_rc_scope_tensor (fun () ->
    t_with_training.apply_with_training xs ~is_training)
;;

let id = { apply = Fn.id }
let id_ = { apply_with_training = (fun xs ~is_training:_ -> xs) }
let of_fn apply = { apply }
let of_fn_ apply_with_training = { apply_with_training }

let sequential t_list =
  let apply xs =
    Tensor.with_rc_scope_tensor (fun () ->
      List.fold__local__local t_list ~init:xs ~f:(fun acc t -> t.apply acc))
  in
  { apply }
;;

let sequential_ t_list =
  let apply_with_training xs ~is_training =
    Tensor.with_rc_scope_tensor (fun () ->
      List.fold__local__local t_list ~init:xs ~f:(fun acc t ->
        t.apply_with_training acc ~is_training))
  in
  { apply_with_training }
;;

let embeddings
  ?(sparse = false)
  ?(scale_grad_by_freq = false)
  vs
  ~num_embeddings
  ~embedding_dim
  =
  let weight =
    Var_store.new_var
      vs
      ~shape:[ num_embeddings; embedding_dim ]
      ~init:(Normal { mean = 0.; stdev = 1. })
      ~name:"weight"
  in
  let apply indices =
    Tensor.embedding ~weight ~indices ~padding_idx:(-1) ~sparse ~scale_grad_by_freq
  in
  { apply }
;;

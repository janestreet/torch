open Base
open Torch

(* Replicating:
   https://github.com/pytorch/examples/blob/42068585f7da9a2f8656f793d21c30ee6806c6b9/distributed/tensor_parallelism/llama2_model.py

   Uses the tinyshakespeare dataset which can be downloaded at: *)
let dataset_url =
  "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
;;

let learning_rate = 0.0003
let batch_size = 64
let seq_len = 128
let block_size = 128
let epochs = 100
let sampling_length = 1024
let temperature = 1.0
let train_dataset_filename = "data/input.txt"

type config =
  { dim : int
  ; n_layers : int
  ; n_heads : int
  ; n_kv_heads : int option
  ; vocab_size : int
  ; multiple_of : int
  ; ffn_dim_multiplier : float option
  ; norm_eps : float
  ; max_batch_size : int
  ; max_seq_len : int
  ; (* If `True`, then each transformer block init uses its layer ID, and if `False`, each
       uses the total number of transformer blocks *)
    depth_init : bool
  }

let trunc_normal_ tensor ~mean ~std ~a ~b =
  (* Replicating nn.init.trunc_normal_, implemented in python in torch *)
  let erf x = Tensor.of_float0 x |> Tensor.erf_ |> Tensor.to_float0_exn in
  let norm_cdf x = (1. +. erf (x /. Float.sqrt 2.)) /. 2. in
  let l = norm_cdf ((a -. mean) /. std) in
  let u = norm_cdf ((b -. mean) /. std) in
  let tensor = Tensor.uniform_ tensor ~from:l ~to_:u in
  let tensor = Tensor.erfinv_ tensor in
  let tensor = Tensor.mul_scalar_ tensor (Scalar.float (std *. Float.sqrt 2.)) in
  let tensor = Tensor.add_scalar_ tensor (Scalar.float mean) in
  let tensor = Tensor.clamp_ tensor ~min:(Scalar.float a) ~max:(Scalar.float b) in
  tensor
;;

let linear_layer_trunc_normal_init
  ?use_bias
  ?(a = -2.)
  ?(b = 2.)
  vs
  ~input_dim
  ~output_dim
  ~mean
  ~std
  =
  let weight_shape = [ output_dim; input_dim ] in
  let weight_tensor = Tensor.zeros weight_shape ~kind:(T Float) in
  let weight_tensor = trunc_normal_ weight_tensor ~mean ~std ~a ~b in
  Layer.linear vs ~input_dim output_dim ?use_bias ~w_init:(Copy weight_tensor)
;;

let precompute_freqs_cis ?(theta = 10000.0) vs ~dim ~end_ =
  let device = Var_store.device vs in
  let freqs =
    Tensor.pow
      (Tensor.of_float0 theta)
      ~exponent:
        (Tensor.( / )
           (Tensor.arange_start_step
              ()
              ~options:(T Float, device)
              ~start:(Scalar.int 0)
              ~end_:(Scalar.int dim)
              ~step:(Scalar.int 2)
            |> Tensor.slice ~dim:0 ~start:(Some 0) ~end_:(Some (dim / 2)) ~step:1)
           (Tensor.of_float0 (Float.of_int dim)))
  in
  let freqs = Tensor.( / ) (Tensor.of_float0 1.0) freqs in
  let t = Tensor.arange ~end_:(Scalar.int end_) ~options:(T Float, device) in
  let freqs = Tensor.outer t ~vec2:freqs |> Tensor.to_type ~type_:(T Float) in
  let freqs_cis = Tensor.polar ~abs:(Tensor.ones_like freqs) ~angle:freqs in
  freqs_cis
;;

let reshape_for_broadcast ~freqs_cis ~x =
  let ndim = Tensor.ndim x in
  assert (1 < ndim);
  let desired_shape =
    [ List.nth_exn (Tensor.shape x) 1; List.last_exn (Tensor.shape x) ]
  in
  let actual_shape = Tensor.shape freqs_cis in
  assert (List.equal Int.equal actual_shape desired_shape);
  let shape =
    List.mapi (Tensor.shape x) ~f:(fun i d -> if i = 1 || i = ndim - 1 then d else 1)
  in
  Tensor.view freqs_cis ~size:shape
;;

let apply_rotary_emb ~xq ~xk ~freqs_cis =
  let xq_ =
    xq
    |> Tensor.to_type ~type_:(T Float)
    |> Tensor.reshape ~shape:(List.drop_last_exn (Tensor.shape xq) @ [ -1; 2 ])
    |> Tensor.view_as_complex
  and xk_ =
    xk
    |> Tensor.to_type ~type_:(T Float)
    |> Tensor.reshape ~shape:(List.drop_last_exn (Tensor.shape xk) @ [ -1; 2 ])
    |> Tensor.view_as_complex
  in
  let freqs_cis = reshape_for_broadcast ~freqs_cis ~x:xq_ in
  let xq_out =
    xq_ |> Tensor.( * ) freqs_cis |> Tensor.view_as_real |> Tensor.flatten ~start_dim:3
  and xk_out =
    xk_ |> Tensor.( * ) freqs_cis |> Tensor.view_as_real |> Tensor.flatten ~start_dim:3
  in
  xq_out |> Tensor.type_as xq, xk_out |> Tensor.type_as xk
;;

let repeat_kv ~x ~n_rep =
  let bs, slen, n_kv_heads, head_dim = Tensor.shape4_exn x in
  if n_rep = 1
  then x
  else
    x
    |> Tensor.unsqueeze ~dim:(-2)
    |> Tensor.expand ~size:[ bs; slen; n_kv_heads; n_rep; head_dim ] ~implicit:false
    |> Tensor.reshape ~shape:[ bs; slen; n_kv_heads * n_rep; head_dim ]
;;

let rms_norm ?(eps = 1e-6) vs ~dim =
  let weight =
    Var_store.new_var ~trainable:true vs ~shape:[ dim ] ~init:Ones ~name:"weight"
  in
  let norm x =
    Tensor.pow x ~exponent:(Tensor.of_float0 2.0)
    |> Tensor.mean_dim ~dim:(Some [ -1 ]) ~keepdim:true ~dtype:(T Float)
    |> Tensor.( + ) (Tensor.of_float0 eps)
    |> Tensor.rsqrt
    |> Tensor.( * ) x
  in
  Layer.of_fn (fun xs -> norm xs |> Tensor.( * ) weight)
;;

let attention vs cfg ~weight_init_std =
  let n_heads = cfg.n_heads in
  let n_kv_heads = Option.value cfg.n_kv_heads ~default:cfg.n_heads in
  let n_rep = n_heads / n_kv_heads in
  let head_dim = cfg.dim / n_heads in
  let wq =
    linear_layer_trunc_normal_init
      vs
      ~input_dim:cfg.dim
      ~output_dim:(cfg.n_heads * head_dim)
      ~use_bias:false
      ~mean:0.0
      ~std:0.02
  in
  let wk =
    linear_layer_trunc_normal_init
      vs
      ~input_dim:cfg.dim
      ~output_dim:(n_kv_heads * head_dim)
      ~use_bias:false
      ~mean:0.0
      ~std:0.02
  in
  let wv =
    linear_layer_trunc_normal_init
      vs
      ~input_dim:cfg.dim
      ~output_dim:(n_kv_heads * head_dim)
      ~use_bias:false
      ~mean:0.0
      ~std:0.02
  in
  let wo =
    linear_layer_trunc_normal_init
      vs
      ~input_dim:(cfg.n_heads * head_dim)
      ~output_dim:cfg.dim
      ~use_bias:false
      ~mean:0.0
      ~std:weight_init_std
  in
  fun ~freqs_cis x ->
    let bsz, seqlen, _ = Tensor.shape3_exn x in
    let xq = Layer.forward wq x
    and xk = Layer.forward wk x
    and xv = Layer.forward wv x in
    let xq = Tensor.view xq ~size:[ bsz; seqlen; n_heads; head_dim ]
    and xk = Tensor.view xk ~size:[ bsz; seqlen; n_kv_heads; head_dim ]
    and xv = Tensor.view xv ~size:[ bsz; seqlen; n_kv_heads; head_dim ] in
    let xq, xk = apply_rotary_emb ~xq ~xk ~freqs_cis in
    let keys = repeat_kv ~x:xk ~n_rep
    and values = repeat_kv ~x:xv ~n_rep in
    let xq = Tensor.transpose xq ~dim0:1 ~dim1:2
    and xk = Tensor.transpose keys ~dim0:1 ~dim1:2
    and xv = Tensor.transpose values ~dim0:1 ~dim1:2 in
    let output =
      Tensor.scaled_dot_product_attention
        ~query:xq
        ~key:xk
        ~value:xv
        ~attn_mask:None
        ~dropout_p:0.0
        ~is_causal:true
        ~scale:None
        ~enable_gqa:false
      |> Tensor.transpose ~dim0:1 ~dim1:2
      |> Tensor.contiguous
      |> Tensor.view ~size:[ bsz; seqlen; -1 ]
    in
    Layer.forward wo output
;;

let feed_forward
  ?(ffn_dim_multiplier = 1.)
  vs
  ~dim
  ~hidden_dim
  ~multiple_of
  ~weight_init_std
  =
  let hidden_dim = 2 * hidden_dim / 3 in
  let hidden_dim = ffn_dim_multiplier *. Float.of_int hidden_dim |> Int.of_float in
  let hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of) in
  let w1 =
    linear_layer_trunc_normal_init
      vs
      ~input_dim:dim
      ~output_dim:hidden_dim
      ~use_bias:false
      ~mean:0.0
      ~std:0.02
  in
  let w2 =
    linear_layer_trunc_normal_init
      vs
      ~input_dim:hidden_dim
      ~output_dim:dim
      ~use_bias:false
      ~mean:0.0
      ~std:weight_init_std
  in
  let w3 =
    linear_layer_trunc_normal_init
      vs
      ~input_dim:dim
      ~output_dim:hidden_dim
      ~use_bias:false
      ~mean:0.0
      ~std:weight_init_std
  in
  Layer.of_fn (fun x ->
    Layer.forward
      w2
      (Tensor.( * ) (Tensor.silu (Layer.forward w1 x)) (Layer.forward w3 x)))
;;

let transformer_block vs ~model_args ~layer_idx =
  let weight_init_std =
    let index = if model_args.depth_init then layer_idx + 1 else model_args.n_layers in
    0.02 /. ((2. *. Float.of_int index) **. 0.5)
  in
  let attention = attention vs model_args ~weight_init_std
  and feed_forward =
    feed_forward
      vs
      ~dim:model_args.dim
      ~hidden_dim:(4 * model_args.dim)
      ~multiple_of:model_args.multiple_of
      ~weight_init_std
      ?ffn_dim_multiplier:model_args.ffn_dim_multiplier
  and attention_norm = rms_norm vs ~dim:model_args.dim ~eps:model_args.norm_eps
  and ffn_norm = rms_norm vs ~dim:model_args.dim ~eps:model_args.norm_eps in
  fun ~freqs_cis x ->
    let h = Tensor.( + ) x (Layer.forward attention_norm x |> attention ~freqs_cis) in
    Tensor.( + ) h (Layer.forward ffn_norm h |> Layer.forward feed_forward)
;;

let transformer vs ~model_args =
  let precomputed_freqs_cis =
    precompute_freqs_cis
      vs
      ~dim:(model_args.dim / model_args.n_heads)
      ~end_:(model_args.max_seq_len * 2)
  in
  let tok_embeddings =
    Layer.embeddings
      vs
      ~num_embeddings:model_args.vocab_size
      ~embedding_dim:model_args.dim
  in
  let freqs_cis =
    Var_store.new_var_copy
      vs
      ~trainable:false
      ~name:"freqs_cis"
      ~src:precomputed_freqs_cis
  in
  let layers =
    List.init model_args.n_layers ~f:(fun layer_idx ->
      transformer_block
        Var_store.(vs / [%string "layer%{layer_idx#Int}"])
        ~model_args
        ~layer_idx)
  in
  let norm =
    rms_norm Var_store.(vs / "rms_norm") ~dim:model_args.dim ~eps:model_args.norm_eps
  in
  let final_out_std = Float.of_int model_args.dim **. -0.5 in
  let cutoff_factor = 3. in
  let output =
    linear_layer_trunc_normal_init
      Var_store.(vs / "output")
      ~use_bias:false
      ~input_dim:model_args.dim
      ~output_dim:model_args.vocab_size
      ~mean:0.0
      ~std:final_out_std
      ~a:(-.cutoff_factor *. final_out_std)
      ~b:(cutoff_factor *. final_out_std)
  in
  Layer.of_fn (fun tokens ->
    let (_ : int), seqlen = Tensor.shape2_exn tokens in
    let h = Layer.forward tok_embeddings tokens in
    let freqs_cis =
      Tensor.slice freqs_cis ~dim:0 ~start:(Some 0) ~end_:(Some seqlen) ~step:1
    in
    let h = List.fold layers ~init:h ~f:(fun h layer -> (layer ~freqs_cis) h) in
    let h = Layer.forward norm h in
    let output = Layer.forward output h |> Tensor.to_type ~type_:(T Float) in
    output)
;;

let sample ~transformer ~dataset ~device =
  let sample_start_time = Core.Time_ns.now () in
  let input = Tensor.zeros [ 1; block_size ] ~kind:(T Int64) ~device in
  let output =
    List.init sampling_length ~f:Fn.id
    |> List.fold_map ~init:input ~f:(fun input _idx ->
      Stdlib.Gc.full_major ();
      let logits = Layer.forward transformer input |> Tensor.select ~dim:1 ~index:(-1) in
      let logits = Tensor.(logits / f temperature) in
      let sampled_y =
        Tensor.softmax logits ~dim:(-1) ~dtype:(T Float)
        |> Tensor.multinomial ~num_samples:1 ~replacement:true
      in
      let input =
        Tensor.cat [ input; Tensor.view sampled_y ~size:[ 1; 1 ] ] ~dim:1
        |> Tensor.narrow ~dim:1 ~start:1 ~length:block_size
      in
      input, sampled_y)
    |> snd
    |> List.map ~f:(fun sampled_y ->
      Text_helper.char dataset ~label:(Tensor.int_value sampled_y))
    |> String.of_char_list
  in
  let sample_elapsed_time_ms =
    Core.Time_ns.diff (Core.Time_ns.now ()) sample_start_time |> Core.Time_ns.Span.to_ms
  in
  Stdio.printf "sampling took %f ms\n" sample_elapsed_time_ms;
  output
;;

let train vs ~transformer ~dataset =
  let device = Var_store.device vs in
  let labels = Text_helper.labels dataset in
  let adam = Optimizer.adam vs ~learning_rate in
  let batches_per_epoch = (Text_helper.total_length dataset - seq_len) / batch_size in
  Checkpointing.loop
    ~start_index:1
    ~end_index:epochs
    ~var_stores:[ vs ]
    ~checkpoint_base:"llama2.ot"
    ~checkpoint_every:(`iters 1)
    (fun ~index:epoch_idx ->
       Stdio.Out_channel.write_all
         (Printf.sprintf "out.txt.%d" epoch_idx)
         ~data:(sample ~transformer ~dataset ~device);
       let start_time = Unix.gettimeofday () in
       let sum_loss = ref 0. in
       Text_helper.iter dataset ~device ~batch_size ~seq_len ~f:(fun batch_idx ~xs ~ys ->
         let logits = Layer.forward transformer xs in
         (* Compute the cross-entropy loss. *)
         let loss =
           Tensor.cross_entropy_for_logits
             (Tensor.view logits ~size:[ batch_size * seq_len; labels ])
             ~targets:(Tensor.view ys ~size:[ batch_size * seq_len ])
         in
         sum_loss := !sum_loss +. Tensor.float_value loss;
         Stdio.printf
           "%d/%d %f\r%!"
           batch_idx
           batches_per_epoch
           (!sum_loss /. Float.of_int (1 + batch_idx));
         Optimizer.backward_step ~clip_grad:(Norm2 4.) adam ~loss;
         if batch_idx % 100 = 0
         then Stdio.printf "%s\n" (sample ~transformer ~dataset ~device));
       Stdio.printf
         "%d %.0fs %f\n%!"
         epoch_idx
         (Unix.gettimeofday () -. start_time)
         (!sum_loss /. Float.of_int batches_per_epoch))
;;

let () =
  let device = Device.cuda_if_available () in
  if not (Stdlib.Sys.file_exists train_dataset_filename)
  then
    raise_s
      (Sexp.message
         "Dataset not found, try downloading from dataset_url"
         [ "expected_dataset_filepath", sexp_of_string train_dataset_filename
         ; "dataset_url", sexp_of_string dataset_url
         ]);
  let dataset = Text_helper.create ~filename:train_dataset_filename in
  let vs = Var_store.create ~name:"llama2" ~device () in
  let labels = Text_helper.labels dataset in
  Stdio.printf
    "Dataset loaded, length: %d, labels: %d.\n%!"
    (Text_helper.total_length dataset)
    labels;
  let model_args =
    { dim = 128
    ; n_layers = 4
    ; n_heads = 4
    ; n_kv_heads = None
    ; vocab_size = 65
    ; multiple_of = 256
    ; ffn_dim_multiplier = None
    ; norm_eps = 1e-5
    ; max_batch_size = 32
    ; max_seq_len = 4096
    ; depth_init = true
    }
  in
  let transformer = transformer vs ~model_args in
  match Stdlib.Sys.argv with
  | [| _bin |] | [| _bin; "train" |] -> train vs ~transformer ~dataset
  | [| _bin; "sample"; filename |] ->
    let named_tensors = Var_store.all_vars vs in
    Serialize.load_multi_ ~named_tensors ~filename;
    sample ~transformer ~dataset ~device |> Stdio.print_endline
  | _ -> failwith "usage: llama2 (train|sample weight.ot)"
;;

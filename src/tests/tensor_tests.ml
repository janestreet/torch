open Base
open Sexplib.Conv
open Torch

let%expect_test "addition" =
  let t = Tensor.(f 41. + f 1.) in
  Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn t);
  [%expect {| 42 |}];
  let t = Tensor.float_vec [ 1.; 42.; 1337. ] in
  Stdio.printf !"%{sexp:float array}\n" Tensor.(to_float1_exn (t * t));
  [%expect {| (1 1764 1787569) |}];
  Stdio.printf !"%{sexp:float array}\n" Tensor.(to_float1_exn (t + f 1.5));
  [%expect {| (2.5 43.5 1338.5) |}]
;;

let%expect_test _ =
  let open Tensor in
  let t = zeros [ 4; 2 ] in
  t.%.{[ 1; 1 ]} <- 42.0;
  t.%.{[ 3; 0 ]} <- 1.337;
  for i = 0 to 3 do
    Stdio.printf "%f %f\n" t.%.{[ i; 0 ]} t.%.{[ i; 1 ]}
  done;
  [%expect
    {|
    0.000000 0.000000
    0.000000 42.000000
    0.000000 0.000000
    1.337000 0.000000
    |}]
;;

let%expect_test _ =
  let open Tensor in
  let t = zeros [ 5; 2 ] in
  t += f 1.;
  narrow t ~dim:0 ~start:1 ~length:3 += f 2.;
  narrow t ~dim:1 ~start:1 ~length:1 -= f 3.;
  Stdio.printf !"%{sexp:float array array}\n" (Tensor.to_float2_exn t);
  [%expect {| ((1 -2) (3 0) (3 0) (3 0) (1 -2)) |}]
;;

let%expect_test _ =
  let t = List.init 5 ~f:Float.of_int |> Tensor.float_vec in
  let array = Tensor.to_float1_exn t in
  let array_narrow = Tensor.narrow t ~dim:0 ~start:1 ~length:3 |> Tensor.to_float1_exn in
  Stdio.printf !"%{sexp:float array}\n" array;
  Stdio.printf !"%{sexp:float array}\n" array_narrow;
  [%expect
    {|
    (0 1 2 3 4)
    (1 2 3)
    |}]
;;

let%expect_test _ =
  let t = Tensor.of_int2 [| [| 3; 4; 5 |]; [| 2; 3; 4 |] |] in
  Tensor.(narrow t ~dim:1 ~start:0 ~length:1 += of_int0 42);
  Stdio.printf !"%{sexp:int array array}\n" (Tensor.to_int2_exn t);
  [%expect {| ((45 4 5) (44 3 4)) |}]
;;

let%expect_test _ =
  let t = Tensor.zeros [ 2; 3; 2 ] in
  let u = Tensor.narrow t ~dim:1 ~start:1 ~length:2 in
  let v = Tensor.get u 1 in
  let w1 = Tensor.copy v in
  let w2 = Tensor.copy v in
  Tensor.(w1 += f 1.);
  Stdio.printf !"%{sexp:float array array}\n" (Tensor.to_float2_exn w1);
  [%expect {| ((1 1) (1 1)) |}];
  Stdio.printf !"%{sexp:float array array}\n" (Tensor.to_float2_exn w2);
  [%expect {| ((0 0) (0 0)) |}];
  Stdio.printf !"%{sexp:float array array array}\n" (Tensor.to_float3_exn t);
  [%expect {| (((0 0) (0 0) (0 0)) ((0 0) (0 0) (0 0))) |}];
  Tensor.(v += f 1.);
  Stdio.printf !"%{sexp:float array array array}\n" (Tensor.to_float3_exn t);
  [%expect {| (((0 0) (0 0) (0 0)) ((0 0) (1 1) (1 1))) |}]
;;

let%expect_test "copy_to_bigstring works" =
  (* not just any string will do *)
  let initial_bytes = " (^._.^)__/ " |> String.to_array in
  let n_bytes = Array.length initial_bytes in
  let src =
    Bigarray.Array1.of_array Bigarray.char Bigarray.c_layout initial_bytes
    |> Bigarray.genarray_of_array1
    |> Tensor.of_bigarray
  in
  let dst = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n_bytes in
  Tensor.copy_to_bigstring ~src ~dst ~dst_pos:0 ~dst_len:n_bytes;
  let dst_array = Array.init n_bytes ~f:(fun _ -> Char.of_int_exn 0) in
  for i = 0 to n_bytes - 1 do
    Array.set dst_array i (Bigarray.Array1.get dst i)
  done;
  Stdio.print_endline (String.of_array dst_array);
  [%expect {| (^._.^)__/ |}];
  (* demonstrate that it works with a more complicated tensor and puts the floats in
     the right order *)
  let col1 = Tensor.of_float1 [| 1.; 3. |] in
  let col2 = Tensor.of_float1 [| 2.; 4. |] in
  let src = Tensor.stack [ col1; col2 ] ~dim:1 in
  let dst = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 16 in
  Tensor.copy_to_bigstring ~src ~dst ~dst_pos:0 ~dst_len:16;
  for i = 0 to Bigarray.Array1.dim dst - 1 do
    Stdio.printf "%02x" (Char.to_int (Bigarray.Array1.get dst i))
  done;
  (* this is what we expect:
     >>> base64.b16encode(struct.pack("ffff", 1., 2., 3., 4.)).lower().decode("ascii")
     '0000803f000000400000404000008040'
  *)
  [%expect {| 0000803f000000400000404000008040 |}]
;;

let%expect_test "copy_to_bigstring validates pos/len and fails cleanly" =
  let tensor = Tensor.of_int1 [| 0x2323232323232323; 0x2424242424242424 |] in
  let dst = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 16 in
  (* this works *)
  Tensor.copy_to_bigstring ~src:tensor ~dst ~dst_pos:0 ~dst_len:16;
  for i = 0 to Bigarray.Array1.dim dst - 1 do
    Stdio.printf "%c" (Bigarray.Array1.get dst i)
  done;
  [%expect {| ########$$$$$$$$ |}];
  (* this also works *)
  let dst = Bigarray.Array1.init Bigarray.char Bigarray.c_layout 20 (fun _ -> '.') in
  Tensor.copy_to_bigstring ~src:tensor ~dst ~dst_pos:0 ~dst_len:16;
  for i = 0 to Bigarray.Array1.dim dst - 1 do
    Stdio.printf "%c" (Bigarray.Array1.get dst i)
  done;
  [%expect {| ########$$$$$$$$.... |}];
  (* these are correctly rejected *)
  Expect_test_helpers_base.show_raise (fun () : unit ->
    Tensor.copy_to_bigstring ~src:tensor ~dst ~dst_pos:10 ~dst_len:16);
  [%expect {| (raised (Invalid_argument "pos + len past end: 10 + 16 > 20")) |}];
  Expect_test_helpers_base.show_raise (fun () : unit ->
    Tensor.copy_to_bigstring ~src:tensor ~dst ~dst_pos:10 ~dst_len:10);
  [%expect
    {| (raised (Failure "bytes is not the correct length for this tensor: 16 != 10")) |}];
  Expect_test_helpers_base.show_raise (fun () : unit ->
    Tensor.copy_to_bigstring ~src:tensor ~dst ~dst_pos:2 ~dst_len:5);
  [%expect
    {| (raised (Failure "bytes is not the correct length for this tensor: 16 != 5")) |}];
  Expect_test_helpers_base.show_raise (fun () : unit ->
    Tensor.copy_to_bigstring ~src:tensor ~dst ~dst_pos:(-1) ~dst_len:16);
  [%expect {| (raised (Invalid_argument "Negative position: -1")) |}];
  (* these are completely illegal, and crucially don't segv *)
  Expect_test_helpers_base.show_raise (fun () : unit ->
    Tensor.copy_to_bigstring ~src:tensor ~dst ~dst_pos:(-1) ~dst_len:16);
  [%expect {| (raised (Invalid_argument "Negative position: -1")) |}];
  Expect_test_helpers_base.show_raise (fun () : unit ->
    Tensor.copy_to_bigstring ~src:tensor ~dst ~dst_pos:100 ~dst_len:16);
  [%expect {| (raised (Invalid_argument "pos + len past end: 100 + 16 > 20")) |}]
;;

let%expect_test "copy_from_bigstring works" =
  let bigstring_of_string str =
    Bigarray.Array1.init
      Bigarray.char
      Bigarray.c_layout
      (String.length str)
      (String.get str)
  in
  let src = bigstring_of_string "\x01\x00\x00\x00\x02\x00\x00\x00" in
  let dst = Tensor.zeros ~kind:(T Int) [ 2 ] in
  Tensor.copy_from_bigstring ~src ~src_pos:0 ~src_len:8 ~dst;
  Stdio.printf !"%{sexp:int array}\n" (Tensor.to_int1_exn dst);
  [%expect {| (1 2) |}];
  (* demonstrate that it works with a more complicated tensor (which just gets
     overwritten) *)
  let col1 = Tensor.zeros ~kind:(T Int) [ 2 ] in
  let col2 = Tensor.zeros ~kind:(T Int) [ 2 ] in
  let src =
    bigstring_of_string "\x05\x00\x00\x00\x06\x00\x00\x00\x07\x00\x00\x00\x08\x00\x00\x00"
  in
  let dst = Tensor.stack [ col1; col2 ] ~dim:1 in
  Tensor.copy_from_bigstring ~src ~src_pos:0 ~src_len:16 ~dst;
  Stdio.printf !"%{sexp:int array array}\n" (Tensor.to_int2_exn dst);
  [%expect {| ((5 6) (7 8)) |}]
;;

let%expect_test "copy_from_bigstring validates pos/len and fails cleanly" =
  let bytes = "\xff\xff\x01\x00\x00\x00\x02\x00\x00\x00" in
  let src =
    Bigarray.Array1.init
      Bigarray.char
      Bigarray.c_layout
      (String.length bytes)
      (String.get bytes)
  in
  let dst = Tensor.zeros ~kind:(T Int) [ 2 ] in
  (* this works fine *)
  Tensor.copy_from_bigstring ~src ~src_pos:2 ~src_len:8 ~dst;
  Stdio.printf !"%{sexp:int array}\n" (Tensor.to_int1_exn dst);
  [%expect {| (1 2) |}];
  (* these are correctly rejected (and crucially don't segv) *)
  Expect_test_helpers_base.show_raise (fun () : unit ->
    Tensor.copy_from_bigstring ~src ~src_pos:1 ~src_len:9 ~dst);
  [%expect
    {| (raised (Failure "bytes is not the correct length for this tensor: 8 != 9")) |}];
  Expect_test_helpers_base.show_raise (fun () : unit ->
    Tensor.copy_from_bigstring ~src ~src_pos:5 ~src_len:5 ~dst);
  [%expect
    {| (raised (Failure "bytes is not the correct length for this tensor: 8 != 5")) |}];
  Expect_test_helpers_base.show_raise (fun () : unit ->
    Tensor.copy_from_bigstring ~src ~src_pos:5 ~src_len:8 ~dst);
  [%expect {| (raised (Invalid_argument "pos + len past end: 5 + 8 > 10")) |}];
  Expect_test_helpers_base.show_raise (fun () : unit ->
    Tensor.copy_from_bigstring ~src ~src_pos:16 ~src_len:8 ~dst);
  [%expect {| (raised (Invalid_argument "pos + len past end: 16 + 8 > 10")) |}];
  Expect_test_helpers_base.show_raise (fun () : unit ->
    Tensor.copy_from_bigstring ~src ~src_pos:(-1) ~src_len:8 ~dst);
  [%expect {| (raised (Invalid_argument "Negative position: -1")) |}]
;;

let%expect_test _ =
  let logits = Tensor.of_float1 [| -1.; 0.5; 0.25; 0.; 2.; 4.; -1. |] in
  let eval_and_print ~target =
    let bce1 =
      Tensor.(bce_loss (sigmoid logits) ~targets:(ones_like logits * f target))
    in
    let bce2 =
      Tensor.(bce_loss_with_logits logits ~targets:(ones_like logits * f target))
    in
    let bce3 =
      Tensor.(
        (-f target * log (sigmoid logits))
        - ((f 1. - f target) * log (f 1. - sigmoid logits)))
      |> Tensor.mean
    in
    Stdio.printf
      !"%{sexp:float} %{sexp:float} %{sexp:float}\n"
      (Tensor.to_float0_exn bce1)
      (Tensor.to_float0_exn bce2)
      (Tensor.to_float0_exn bce3)
  in
  eval_and_print ~target:0.;
  [%expect {| 1.3235375881195068 1.3235378265380859 1.3235375881195068 |}];
  eval_and_print ~target:0.5;
  [%expect {| 0.98425191640853882 0.98425203561782837 0.98425191640853882 |}];
  eval_and_print ~target:1.;
  [%expect {| 0.64496642351150513 0.64496642351150513 0.64496642351150513 |}]
;;

let%expect_test _ =
  let vs = Tensor.of_float1 [| -1.01; -1.; -0.99; 0.5; 0.25; 0.; 2.; 4.; -1.; -3. |] in
  Stdio.printf
    !"%{sexp:float array} %{sexp:float}\n"
    Tensor.(huber_loss vs (Tensor.f 0.) ~reduction:None |> to_float1_exn)
    Tensor.(huber_loss vs (Tensor.f 0.) |> to_float0_exn);
  [%expect
    {| (0.50999999046325684 0.5 0.49005001783370972 0.125 0.03125 0 1.5 3.5 0.5 2.5) 0.9656299352645874 |}]
;;

let%expect_test _ =
  let vs = List.range 1 10 |> Array.of_list |> Tensor.of_int1 in
  let chunk = Tensor.chunk vs ~chunks:4 ~dim:0 in
  Stdio.printf
    !"%{sexp:int array} %{sexp:int array list}\n"
    (Tensor.to_int1_exn vs)
    (List.map chunk ~f:Tensor.to_int1_exn);
  [%expect {| (1 2 3 4 5 6 7 8 9) ((1 2 3) (4 5 6) (7 8 9)) |}]
;;

let%expect_test _ =
  let vs = Tensor.of_int1 [| 3; 1; 4 |] in
  let ws = Tensor.to_type vs ~type_:(T Float) in
  let xs = Tensor.reshape vs ~shape:[ -1; 1 ] in
  Stdio.printf
    "%b %b %b %b\n"
    (Tensor.eq vs vs)
    (Tensor.eq vs ws)
    (Tensor.eq ws ws)
    (Tensor.eq vs xs);
  [%expect {| true false true false |}];
  let ws = Tensor.of_int1 [| 3; 1 |] in
  let xs = Tensor.of_int1 [| 4; 2; 5 |] in
  Stdio.printf "%b %b\n" (Tensor.eq vs ws) (Tensor.eq vs xs);
  [%expect {| false false |}];
  Tensor.(xs -= of_int0 1);
  Stdio.printf "%b %b\n" (Tensor.eq vs ws) (Tensor.eq vs xs);
  [%expect {| false true |}]
;;

let%expect_test _ =
  let t = Tensor.of_int2 [| [| 3; 1; 4 |]; [| 1; 5; 9 |] |] in
  Tensor.to_list t
  |> List.iter ~f:(fun t -> Tensor.to_int1_exn t |> Stdio.printf !"%{sexp:int array}\n");
  [%expect
    {|
    (3 1 4)
    (1 5 9)
    |}];
  assert (Tensor.device t = Cpu)
;;

let%expect_test _ =
  (* Element-wise squaring of a vector. *)
  let t = Tensor.of_float1 [| 1.; 2.; 3. |] in
  let t = Tensor.einsum ~equation:"i, i -> i" ~path:None [ t; t ] in
  Stdio.printf !"%{sexp:float array}\n" Tensor.(to_float1_exn t);
  (* Matrix transpose *)
  let t = Tensor.of_int2 [| [| 3; 1; 4 |]; [| 1; 5; 9 |] |] in
  let t = Tensor.einsum ~equation:"ij -> ji" ~path:None [ t ] in
  Tensor.to_list t
  |> List.iter ~f:(fun t -> Tensor.to_int1_exn t |> Stdio.printf !"%{sexp:int array}\n");
  (* Sum all elements *)
  let t = Tensor.einsum ~equation:"ij -> " ~path:None [ t ] in
  Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn t);
  [%expect
    {|
    (1 4 9)
    (3 1)
    (1 5)
    (4 9)
    23
    |}]
;;

let%expect_test _ =
  (* A function returning bool *)
  let float_t = Tensor.of_float1 [| 1. |] in
  let int_t = Tensor.of_int1 [| 1 |] in
  Stdio.printf !"%{sexp:bool}\n" (Torch.Tensor.is_floating_point float_t);
  Stdio.printf !"%{sexp:bool}\n" (Torch.Tensor.is_floating_point int_t);
  [%expect
    {|
    true
    false
    |}]
;;

let%expect_test _ =
  (* A function returning int *)
  let t = Tensor.of_float0 1. in
  Stdio.printf !"%{sexp:int64}\n" (Torch.Tensor._version t);
  let (_ : Tensor.t) = Torch.Tensor.add_ t t in
  Stdio.printf !"%{sexp:int64}\n" (Torch.Tensor._version t);
  [%expect
    {|
    0
    1
    |}]
;;

let%expect_test _ =
  let dst = Tensor.rand [ 10 ] in
  Stdio.printf !"%{sexp:int list}\n" (Torch.Tensor.shape dst);
  [%expect {| (10) |}];
  let src = Tensor.rand [ 20; 10 ] in
  Tensor.set_data dst ~src;
  Stdio.printf !"%{sexp:int list}\n" (Torch.Tensor.shape dst);
  [%expect {| (20 10) |}]
;;

let%expect_test "argmax" =
  let t = Tensor.of_float2 [| [| 10.; 20.; 40. |]; [| 60.; 50.; 30. |] |] in
  let along_dim0 = Tensor.argmax t ~dim:0 in
  Stdio.printf !"%{sexp: int array}\n" (Tensor.to_int1_exn along_dim0);
  [%expect {| (1 1 0) |}];
  let along_dim1 = Tensor.argmax t ~dim:1 in
  Stdio.printf !"%{sexp: int array}\n" (Tensor.to_int1_exn along_dim1);
  [%expect {| (2 0) |}]
;;

let%expect_test "nan_to_num" =
  let t = Tensor.of_float1 [| Float.nan; Float.infinity; Float.neg_infinity; 4.0 |] in
  let with_nans_replaced =
    Tensor.nan_to_num ~nan:(Some 1.0) ~posinf:(Some 2.0) ~neginf:(Some 3.0) t
  in
  Stdio.printf !"%{sexp: float array}\n" (Tensor.to_float1_exn with_nans_replaced);
  [%expect {| (1 2 3 4) |}]
;;

let%expect_test "svd" =
  let t = Tensor.of_float2 [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |] in
  let u, s, vt = Tensor.linalg_svd ~a:t ~full_matrices:false ~driver:None in
  Expect_test_helpers_base.with_sexp_round_floats ~significant_digits:6 (fun () ->
    Stdio.printf !"U: %{sexp: float array array}\n" (Tensor.to_float2_exn u);
    Stdio.printf !"Sigma: %{sexp: float array}\n" (Tensor.to_float1_exn s);
    Stdio.printf !"V^T: %{sexp: float array array}\n" (Tensor.to_float2_exn vt));
  [%expect
    {|
    U: ((-0.404554 -0.914514) (-0.914514 0.404554))
    Sigma: (5.46499 0.365966)
    V^T: ((-0.576048 -0.817416) (0.817416 -0.576048))
    |}]
;;

let%expect_test "flatten" =
  let t =
    Tensor.of_float3
      [| [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |]; [| [| 5.0; 6.0 |]; [| 7.0; 8.0 |] |] |]
  in
  Stdio.printf
    !"flatten (0, ...): %{sexp: float array}\n"
    (Tensor.flatten t ~start_dim:0 |> Tensor.to_float1_exn);
  [%expect {| flatten (0, ...): (1 2 3 4 5 6 7 8) |}];
  Stdio.printf
    !"flatten (1, ...): %{sexp: float array array}\n"
    (Tensor.flatten t ~start_dim:1 |> Tensor.to_float2_exn);
  [%expect {| flatten (1, ...): ((1 2 3 4) (5 6 7 8)) |}];
  Stdio.printf
    !"flatten (0, 1): %{sexp: float array array}\n"
    (Tensor.flatten t ~start_dim:0 ~end_dim:1 |> Tensor.to_float2_exn);
  [%expect {| flatten (0, 1): ((1 2) (3 4) (5 6) (7 8)) |}]
;;

let%expect_test "Ensuring optional tensors are passed properly" =
  let query =
    Tensor.of_float3 [| [| [| 1.0; 0.0 |]; [| 0.0; 1.0 |]; [| 1.0; 1.0 |] |] |]
  in
  let key = Tensor.of_float3 [| [| [| 1.0; 0.0 |]; [| 0.0; 1.0 |]; [| 1.0; 1.0 |] |] |] in
  let value =
    Tensor.of_float3 [| [| [| 2.0; 3.0 |]; [| 4.0; 5.0 |]; [| 6.0; 7.0 |] |] |]
  in
  let result =
    Tensor.scaled_dot_product_attention
      ~query
      ~key
      ~value
      ~attn_mask:None
      ~dropout_p:0.0
      ~is_causal:true
      ~scale:None
      ~enable_gqa:false
  in
  (* We will see an error if torch receives is_causal=true and attn_mask != None *)
  Stdio.printf !"Shape: %{sexp: int list}\n" (Tensor.shape result);
  [%expect {| Shape: (1 3 2) |}];
  let attn_mask = Tensor.zeros [ 3; 3 ] in
  let result =
    Tensor.scaled_dot_product_attention
      ~query
      ~key
      ~value
      ~attn_mask:(Some attn_mask)
      ~dropout_p:0.0
      ~is_causal:false
      ~scale:None
      ~enable_gqa:false
  in
  Stdio.printf !"Shape: %{sexp: int list}\n" (Tensor.shape result);
  [%expect {| Shape: (1 3 2) |}]
;;

let%expect_test "precision" =
  let x = 0.5609 in
  let t_float = Tensor.of_float0 x in
  let t_double = Tensor.of_double0 x in
  Stdio.printf !"%.10f\n" Tensor.(exp t_float |> to_float0_exn);
  Stdio.printf !"%.10f\n" Tensor.(exp t_double |> to_float0_exn);
  [%expect
    {|
    1.7522487640
    1.7522488148
    |}];
  Tensor.(t_double.%.{[]} <- 1.2345678901);
  Stdio.printf !"%.10f\n" Tensor.(to_float0_exn t_double);
  [%expect {| 1.2345678901 |}]
;;

let%expect_test "bfloat16" =
  let x = 1.2 in
  let t_float = Tensor.of_float0 x in
  let t_float16 =
    t_float |> Tensor.to_kind ~kind:(T Half) |> Tensor.to_kind ~kind:(T Float)
  in
  let t_bfloat16 =
    t_float |> Tensor.to_kind ~kind:(T BFloat16) |> Tensor.to_kind ~kind:(T Float)
  in
  Stdio.printf !"%.10f\n" Tensor.(t_float16 |> to_float0_exn);
  Stdio.printf !"%.10f\n" Tensor.(t_bfloat16 |> to_float0_exn);
  [%expect
    {|
    1.2001953125
    1.2031250000
    |}]
;;

let%expect_test "gc_test" =
  let open Ctypes in
  let open Torch_bindings.Type_defs in
  let a = Tensor.zeros [ 1; 2; 3 ] in
  let b = Tensor.zeros [ 3; 2; 1 ] in
  Stdlib.Gc.finalise (fun _ -> Stdio.print_endline "freed a") a;
  Stdlib.Gc.finalise (fun _ -> Stdio.print_endline "freed b") b;
  let tensors = [ a; b ] in
  let array =
    (* This is copied from [CArray.of_list] except that there is a custom finaliser so
       we can see when the array goes away. *)
    let arr =
      CArray.make
        ~finalise:(fun _ -> Stdio.print_endline "freed array")
        gc_tensor
        (List.length tensors)
    in
    List.iteri ~f:(CArray.set arr) tensors;
    arr
  in
  (* At this point [array] doesn't actually keep its contents alive. Usually in the
     bindings we'd call into a C++ function now, but if there is a GC first, the
     gc_tensors in the array will be collected. *)
  Stdlib.Gc.full_major ();
  Stdio.print_endline "after gc";
  [%expect {| after gc |}];
  (* Pretend that we called into a C++ function (not necessary for the test) and now
     receive back the results. We'd expect the tensor to still be alive. *)
  let result = CArray.to_list array in
  (* Note: this test is actually still broken. [CArray.to_list] does not increment the
     ref count. Ideally we can put [keep_values_alive] right here, but since the above
     doesn't increment the ref count and there is no good way to do it given the API, we
     have to put [keep_values_alive] at the end. *)
  Stdlib.Gc.full_major ();
  Stdio.print_endline "after second gc";
  List.map result ~f:(Tensor.to_string ~line_size:90)
  |> String.concat ~sep:"\n"
  |> Stdio.print_endline;
  Torch_core.Wrapper_utils.keep_values_alive tensors;
  [%expect
    {|
    freed array
    after second gc
    (1,.,.) =
      0  0  0
      0  0  0
    [ CPUFloatType{1,2,3} ]
    (1,.,.) =
      0
      0

    (2,.,.) =
      0
      0

    (3,.,.) =
      0
      0
    [ CPUFloatType{3,2,1} ]
    |}];
  Stdlib.Gc.full_major ();
  [%expect
    {|
    freed b
    freed a
    |}]
;;

let%expect_test "non-nullable scalar arguments with default values" =
  let t =
    Tensor.arange_start_step
      ~start:(Scalar.int 0)
      ~end_:(Scalar.int 10)
      ~step:(Scalar.int 2)
      ~options:(T Float, Cpu)
      ()
  in
  Stdio.printf !"%{sexp:float array}\n" (Tensor.to_float1_exn t);
  [%expect {| (0 2 4 6 8) |}];
  let t =
    Tensor.arange_start_step
      ~start:(Scalar.int 0)
      ~end_:(Scalar.int 10)
      ~options:(T Float, Cpu)
      ()
  in
  Stdio.printf !"%{sexp:float array}\n" (Tensor.to_float1_exn t);
  [%expect {| (0 1 2 3 4 5 6 7 8 9) |}]
;;

let%expect_test "non-nullable scalar arguments with multiple default values" =
  (* Call [Tensor.softplus] which takes two extra arguments, to make sure they're in the right order *)
  let t =
    Tensor.arange_start_step
      ~start:(Scalar.float (-0.5))
      ~end_:(Scalar.float 1.0)
      ~step:(Scalar.float 0.125)
      ~options:(T Float, Cpu)
      ()
    |> Tensor.softplus ~beta:(Scalar.float 0.9) ~threshold:(Scalar.float 0.5)
  in
  Stdio.printf !"%{sexp:float array}\n" (Tensor.to_float1_exn t);
  [%expect
    {|
    (0.54805439710617065 0.59840929508209229 0.652180016040802
     0.70942044258117676 0.77016353607177734 0.834420382976532 0.902180016040802
     0.97340929508209229 1.0480543375015259 0.625 0.75 0.875)
    |}]
;;

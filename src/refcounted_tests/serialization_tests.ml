open Core
open Torch_refcounted

let write_and_read (tensor @ local) ~print_tensor =
  let filename = Stdlib.Filename.temp_file "torchtest" ".ot" in
  Serialize.save tensor ~filename;
  let y = Serialize.load ~filename in
  let l2 = Tensor.((tensor - y) * (tensor - y)) |> Tensor.sum in
  print_tensor l2;
  Core_unix.unlink filename
;;

let%expect_test _ =
  Tensor.with_rc_scope (fun () ->
    let print_tensor tensor = Stdio.printf "%d\n" (Tensor.to_int0_exn tensor) in
    Tensor.randint ~high:42 ~size:[ 3; 1; 4 ] ~options:(T Int64, Cpu)
    |> write_and_read ~print_tensor;
    [%expect {| 0 |}];
    write_and_read (Tensor.of_int0 1337) ~print_tensor;
    [%expect {| 0 |}])
;;

let%expect_test _ =
  Tensor.with_rc_scope (fun () ->
    let print_tensor tensor = Stdio.printf "%f\n" (Tensor.to_float0_exn tensor) in
    write_and_read (Tensor.randn [ 42; 27 ]) ~print_tensor;
    [%expect {| 0.000000 |}];
    write_and_read (Tensor.of_float0 1337.) ~print_tensor;
    [%expect {| 0.000000 |}])
;;

let write_and_read (named_tensors @ local) =
  let filename = Stdlib.Filename.temp_file "torchtest" ".ot" in
  Serialize.save_multi ~named_tensors ~filename;
  let ys =
    Serialize.load_multi
      ~names:
        (Torch_local_iterators.List.map_local_input named_tensors ~f:(fun (name, _) ->
           String.globalize name))
      ~filename
  in
  Torch_local_iterators.List.iter2_local_exn named_tensors ys ~f:(fun (name, tensor) y ->
    let l2 = Tensor.((tensor - y) * (tensor - y)) |> Tensor.sum in
    let name = String.globalize name in
    (match Tensor.kind l2 with
     | T Int64 -> Stdio.printf "%s %d\n%!" name (Tensor.to_int0_exn l2)
     | T Int -> Stdio.printf "%s %d\n%!" name (Tensor.to_int0_exn l2)
     | T Float -> Stdio.printf "%s %f\n%!" name (Tensor.to_float0_exn l2)
     | T Double -> Stdio.printf "%s %f\n%!" name (Tensor.to_float0_exn l2)
     | _ -> assert false);
    ());
  Core_unix.unlink filename
;;

let%expect_test _ =
  Tensor.with_rc_scope (fun () ->
    write_and_read
      [ "tensor-1", Tensor.of_float1 [| 3.; 14.; 15.; 9265.35 |]
      ; "another", Tensor.of_int0 42
      ; "and yet another", Tensor.of_int2 [| [| 3; -1; -51234 |]; [| 2718; 2818; 28 |] |]
      ] [@nontail]);
  [%expect
    {|
    tensor-1 0.000000
    another 0
    and yet another 0
    |}]
;;

let%expect_test "load_all" =
  Tensor.with_rc_scope (fun () ->
    let filename = Stdlib.Filename.temp_file "torchtest" ".ot" in
    let named_tensors =
      [ "tensor-1", Tensor.of_float1 [| 3.; 14.; 15.; 9265.35 |]
      ; "another", Tensor.of_int0 42
      ; "and yet another", Tensor.of_int2 [| [| 3; -1; -51234 |]; [| 2718; 2818; 28 |] |]
      ]
    in
    Serialize.save_multi ~named_tensors ~filename;
    let ys = Serialize.load_all ~filename in
    Torch_local_iterators.List.iter_local ys ~f:(fun (name, t) ->
      print_endline (String.globalize name);
      Tensor.print t;
      print_endline "");
    Core_unix.unlink filename);
  [%expect
    {|
    tensor-1
        3.0000
       14.0000
       15.0000
     9265.3496
    [ CPUFloatType{4} ]

    another
    42
    [ CPULongType{} ]

    and yet another
         3     -1 -51234
      2718   2818     28
    [ CPULongType{2,3} ]
    |}]
;;

let%expect_test "bigarray" =
  Tensor.with_rc_scope (fun () ->
    let bigarray =
      Bigarray.Array1.of_array Int C_layout [| 5; 6; 7 |] |> Bigarray.genarray_of_array1
    in
    let tensor = Tensor.of_bigarray bigarray in
    Tensor.print tensor;
    [%expect
      {|
       5
       6
       7
      [ CPULongType{3} ]
      |}];
    let t2 = Tensor.of_int1 [| 1; 2; 3 |] in
    Tensor.copy_to_bigarray t2 bigarray;
    let as_list = List.init 3 ~f:(fun i -> Bigarray.Genarray.get bigarray [| i |]) in
    print_s [%sexp (as_list : int list)];
    [%expect {| (1 2 3) |}])
;;

let%expect_test "bigstring" =
  Tensor.with_rc_scope (fun () ->
    let tensor =
      Tensor.of_int2 [| [| 1; 2 |]; [| 3; 4 |] |]
      |> Tensor.to_dtype ~dtype:(T Int8) ~non_blocking:true ~copy:true
    in
    let bigstring = Bigarray.Array1.create Char C_layout 4 in
    Tensor.copy_to_bigstring ~src:tensor ~dst:bigstring ~dst_pos:0 ~dst_len:4;
    let array = Array.init 4 ~f:(Bigarray.Array1.get bigstring) in
    print_s [%sexp (array : char array)];
    [%expect {| ("\001" "\002" "\003" "\004") |}];
    let t2 =
      Tensor.of_int2 [| [| 0; 0 |]; [| 0; 0 |] |]
      |> Tensor.to_dtype ~dtype:(T Int8) ~non_blocking:true ~copy:true
    in
    Tensor.copy_from_bigstring ~src:bigstring ~src_pos:0 ~src_len:4 ~dst:t2;
    Tensor.print t2;
    [%expect
      {|
       1  2
       3  4
      [ CPUCharType{2,2} ]
      |}])
;;

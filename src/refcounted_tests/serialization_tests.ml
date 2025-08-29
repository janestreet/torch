open Base
open Torch_refcounted

let write_and_read tensor ~print_tensor =
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

let write_and_read named_tensors =
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

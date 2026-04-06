open Base
open Sexplib.Conv
open Torch_refcounted

let%expect_test "zero grad" =
  Tensor.with_rc_scope (fun () ->
    (* why we need this cuda check:
       https://github.com/pytorch/pytorch/issues/35736#issuecomment-688078143 *)
    let (_ : bool) = Torch_refcounted.Cuda.is_available () in
    let x = Tensor.f 42. |> Tensor.set_requires_grad ~r:true in
    let y = Tensor.((x * x * x) + (x * x)) in
    Tensor.zero_grad x;
    Tensor.backward y;
    let dy_over_dx = Tensor.grad x in
    Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn dy_over_dx);
    [%expect {| 5376 |}])
;;

let%expect_test "create graph" =
  Tensor.with_rc_scope (fun () ->
    let x = Tensor.f 42. |> Tensor.set_requires_grad ~r:true in
    let y = Tensor.((x * x * x) + (x * x)) in
    Tensor.zero_grad x;
    let dy_over_dx =
      Tensor.run_backward [ y ] [ x ] ~create_graph:true ~keep_graph:true
      |> Torch_local_iterators.List.hd_exn
    in
    Tensor.backward dy_over_dx;
    let dy_over_dx2 = Tensor.grad x in
    Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn dy_over_dx2);
    [%expect {| 254 |}])
;;

let%expect_test "run backward" =
  Tensor.with_rc_scope (fun () ->
    let x = Tensor.f 42. |> Tensor.set_requires_grad ~r:true in
    let y = Tensor.((x * x * x) + (x * x)) in
    let dy_over_dx =
      Tensor.run_backward [ y ] [ x ] |> Torch_local_iterators.List.hd_exn
    in
    Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn dy_over_dx);
    [%expect {| 5376 |}])
;;

let%expect_test "2nd order derivatives" =
  Tensor.with_rc_scope (fun () ->
    let x = Tensor.f 42. |> Tensor.set_requires_grad ~r:true in
    let y = Tensor.((x * x * x) + (x * x)) in
    let dy_over_dx =
      Tensor.run_backward [ y ] [ x ] ~create_graph:true
      |> Torch_local_iterators.List.hd_exn
    in
    Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn dy_over_dx);
    [%expect {| 5376 |}];
    let dy_over_dx2 =
      Tensor.run_backward [ dy_over_dx ] [ x ] |> Torch_local_iterators.List.hd_exn
    in
    Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn dy_over_dx2);
    [%expect {| 254 |}])
;;

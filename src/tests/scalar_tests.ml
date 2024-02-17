open Base
open Torch

let%expect_test _ =
  let x = 42 |> Tensor.Scalar.int |> Tensor.Scalar.to_int in
  Stdio.print !"%d" x;
  [%expect {|
      |}]
;;

let%expect_test _ =
  let x = 42.0 |> Tensor.Scalar.float |> Tensor.Scalar.to_float in
  Stdio.print !"%f" x;
  [%expect {|
      |}]
;;

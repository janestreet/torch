open Torch

let%expect_test _ =
  let x = 42 |> Scalar.int |> Scalar.to_int in
  Stdio.printf !"%d" x;
  [%expect {| 42 |}]
;;

let%expect_test _ =
  let x = 42.0 |> Scalar.float |> Scalar.to_float in
  Stdio.printf !"%f" x;
  [%expect {| 42.000000 |}]
;;

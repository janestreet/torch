open Core
open Torch_refcounted

let%expect_test "Basic text helper usage" =
  let th = Text_helper.create ~filename:"test_data.txt" in
  print_s [%message "" ~total_length:(Text_helper.total_length th : int)];
  [%expect {| (total_length 67) |}];
  print_s [%message "" ~labels:(Text_helper.labels th : int)];
  [%expect {| (labels 21) |}];
  Torch_refcounted_core.Wrapper.manual_seed 42;
  Text_helper.iter th ~seq_len:32 ~batch_size:1 ~f:(fun idx ~xs ~ys ->
    if idx < 2
    then (
      let labels_to_string tensor =
        Tensor.flatten ~start_dim:0 tensor
        |> Tensor.to_int1_exn
        |> Array.map ~f:(fun label -> Text_helper.char th ~label)
        |> String.of_array
      in
      let xs = labels_to_string xs in
      let ys = labels_to_string ys in
      print_s [%message "" ~idx:(idx : int) ~xs:(xs : string) ~ys:(ys : string)]));
  [%expect
    {|
    ((idx 0) (xs "esting the refcounted tensor imp")
     (ys "sting the refcounted tensor impl"))
    ((idx 1) (xs " refcounted tensor implementatio")
     (ys "refcounted tensor implementation"))
    |}]
;;

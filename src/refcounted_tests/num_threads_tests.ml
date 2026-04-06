open Core
open Torch_refcounted_core.Wrapper

let%expect_test "Get and set number of threads" =
  let orig_num_threads = get_num_threads () in
  set_num_threads 1;
  print_s [%message "" ~num_threads:(get_num_threads () : int)];
  [%expect {| (num_threads 1) |}];
  set_num_threads 2;
  print_s [%message "" ~num_threads:(get_num_threads () : int)];
  [%expect {| (num_threads 2) |}];
  set_num_threads orig_num_threads
;;

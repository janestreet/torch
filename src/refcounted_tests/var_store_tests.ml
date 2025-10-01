open Core
open Torch_refcounted

let%expect_test "Creating a tensor in a var store" =
  let vs = Var_store.create () ~name:"my_vars" in
  let gc_t = Var_store.new_var vs ~shape:[ 1 ] ~init:Zeros ~name:"first_var" in
  print_endline (Tensor.For_testing.get_refcount gc_t |> Int.to_string);
  (* Only GC owns a copy, no scope *)
  [%expect {| 1 |}];
  Tensor.print gc_t;
  [%expect
    {|
     0
    [ CPUFloatType{1} ]
    |}]
;;

let%expect_test "Var store tensors are garbage collected" =
  Tensor.with_rc_scope (fun () ->
    let vs = Var_store.create () ~name:"my_vars" in
    let gc_t =
      Var_store.new_var ~trainable:false vs ~shape:[ 1 ] ~init:Zeros ~name:"first_var"
    in
    (* Get an RC copy of the same tensor *)
    let rc_t = Tensor.add_scalar_ gc_t (Scalar.float 1.0) in
    print_endline (Tensor.For_testing.get_refcount rc_t |> Int.to_string);
    [%expect {| 2 |}];
    Gc.keep_alive gc_t;
    Gc.full_major ();
    (* Now the finalizer should have run and decremented refcount by 1 *)
    print_endline (Tensor.For_testing.get_refcount rc_t |> Int.to_string);
    [%expect {| 1 |}])
;;

let%expect_test "Var_store new_var_copy makes a copy" =
  let vs = Var_store.create () ~name:"my_vars" in
  let gc_t =
    Tensor.with_rc_scope (fun () ->
      let rc_t = Tensor.zeros [ 1 ] in
      let gc_t = Var_store.new_var_copy ~trainable:false vs ~src:rc_t ~name:"first_var" in
      (* Two tensors with independent refcounts *)
      print_endline (Tensor.For_testing.get_refcount rc_t |> Int.to_string);
      [%expect {| 1 |}];
      Tensor.For_testing.increment_refcount rc_t;
      print_endline (Tensor.For_testing.get_refcount rc_t |> Int.to_string);
      [%expect {| 2 |}];
      print_endline (Tensor.For_testing.get_refcount gc_t |> Int.to_string);
      [%expect {| 1 |}];
      Tensor.For_testing.decrement_refcount rc_t;
      gc_t)
  in
  print_endline (Tensor.For_testing.get_refcount gc_t |> Int.to_string);
  [%expect {| 1 |}]
;;

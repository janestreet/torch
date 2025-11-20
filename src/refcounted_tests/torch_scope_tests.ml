open Core
open Torch_refcounted

let%expect_test "Refcount for newly created tensor" =
  Tensor.with_rc_scope (fun () ->
    let t = Tensor.zeros [ 1 ] in
    printf "%d\n" (Tensor.For_testing.get_refcount t);
    [%expect {| 1 |}])
;;

let%expect_test "Tensor in inner scope, not returned to outer" =
  Tensor.with_rc_scope (fun () ->
    Tensor.with_rc_scope (fun () ->
      let t = Tensor.zeros [ 1 ] in
      printf "%d\n" (Tensor.For_testing.get_refcount t);
      [%expect {| 1 |}]))
;;

let%expect_test "Tensor from outer scope modified in inner" =
  Tensor.with_rc_scope (fun () ->
    let t = Tensor.ones [ 1 ] in
    printf "%d\n" (Tensor.For_testing.get_refcount t);
    [%expect {| 1 |}];
    Tensor.print t;
    [%expect
      {|
       1
      [ CPUFloatType{1} ]
      |}];
    Tensor.with_rc_scope (fun () ->
      Tensor.( += ) t t;
      printf "%d\n" (Tensor.For_testing.get_refcount t);
      [%expect {| 2 |}];
      Tensor.print t;
      [%expect
        {|
         2
        [ CPUFloatType{1} ]
        |}]);
    printf "%d\n" (Tensor.For_testing.get_refcount t);
    [%expect {| 1 |}];
    Tensor.print t;
    [%expect
      {|
       2
      [ CPUFloatType{1} ]
      |}])
;;

let%expect_test "Tensor returned from inner scope to outer" =
  Tensor.with_rc_scope (fun () ->
    let t_out =
      Tensor.with_rc_scope_tensor (fun () -> exclave_
        let t_inner = Tensor.zeros [ 1 ] in
        printf "%d\n" (Tensor.For_testing.get_refcount t_inner);
        [%expect {| 1 |}];
        t_inner)
    in
    printf "%d\n" (Tensor.For_testing.get_refcount t_out);
    [%expect {| 1 |}])
;;

let%expect_test "Multiple tensors returned from inner scope to outer" =
  Tensor.with_rc_scope (fun () ->
    let t_out =
      Tensor.with_rc_scope_tensors (fun () -> exclave_
        let t_in =
          Tensor.arange_start_step
            ()
            ~step:(Scalar.int (-1))
            ~start:(Scalar.int 5)
            ~end_:(Scalar.int 0)
            ~options:(T Int, Cpu)
        in
        Tensor.print t_in;
        [%expect
          {|
           5
           4
           3
           2
           1
          [ CPUIntType{5} ]
          |}];
        let t_in = Tensor.split t_in ~split_size:1 ~dim:0 in
        let refcounts =
          Torch_local_iterators.List.map_local_input t_in ~f:(fun t ->
            Tensor.For_testing.get_refcount t |> Int.to_string)
        in
        print_endline (String.concat ~sep:" " refcounts);
        [%expect {| 1 1 1 1 1 |}];
        let first = Torch_local_iterators.List.hd_exn t_in in
        Tensor.( += ) first first;
        t_in)
    in
    let refcounts =
      Torch_local_iterators.List.map_local_input t_out ~f:(fun t ->
        Tensor.For_testing.get_refcount t |> Int.to_string)
    in
    print_endline (String.concat ~sep:" " refcounts);
    [%expect {| 1 1 1 1 1 |}];
    let t_concat = Tensor.concat t_out ~dim:0 in
    Tensor.print t_concat;
    [%expect
      {|
       10
        4
        3
        2
        1
      [ CPUIntType{5} ]
      |}])
;;

let%expect_test "Using tensor function without scope stack warns once" =
  Ref.set_temporarily Tensor.warn_on_empty_rc_scope_stack true ~f:(fun () ->
    let t = Tensor.zeros [ 1 ] in
    Tensor.print t;
    [%expect
      {|
      "ocaml-torch: Tried to access the current scope but the scope stack is empty, add a [Tensor.with_rc_scope] around the tensor-related code"
       0
      [ CPUFloatType{1} ]
      |}];
    let t = Tensor.zeros [ 1 ] in
    Tensor.print t;
    [%expect
      {|
       0
      [ CPUFloatType{1} ]
      |}])
;;

let%expect_test "Trying to use with_scope_tensor at the top level warns" =
  Ref.set_temporarily Tensor.warn_on_empty_rc_scope_stack true ~f:(fun () ->
    (try
       let t =
         Tensor.with_rc_scope_tensor (fun () -> exclave_
           (* We should not even create the tensor *)
           [%expect {| |}];
           Tensor.zeros [ 1 ])
       in
       Tensor.print t
     with
     | exn -> print_endline (Exn.to_string exn));
    [%expect
      {|
      "ocaml-torch: Tried to access the current scope but the scope stack is empty, add a [Tensor.with_rc_scope] around the tensor-related code"
       0
      [ CPUFloatType{1} ]
      |}])
;;

let%expect_test "Trying to use with_scope_tensors at the top level warns" =
  Ref.set_temporarily Tensor.warn_on_empty_rc_scope_stack true ~f:(fun () ->
    (try
       let ts =
         Tensor.with_rc_scope_tensors (fun () -> exclave_
           [%expect {| |}];
           [ Tensor.zeros [ 1 ]; Tensor.ones [ 1 ] ])
       in
       Torch_local_iterators.List.iter_local ts ~f:Tensor.print [@nontail]
     with
     | exn -> print_endline (Exn.to_string exn));
    [%expect
      {|
      "ocaml-torch: Tried to access the current scope but the scope stack is empty, add a [Tensor.with_rc_scope] around the tensor-related code"
       0
      [ CPUFloatType{1} ]
       1
      [ CPUFloatType{1} ]
      |}])
;;

let%expect_test "gc multiple times" =
  let t = Tensor.zeros [ 1 ] in
  print_s [%message (Tensor.For_testing.get_refcount t : int)];
  [%expect {| ("Tensor.For_testing.get_refcount t" 1) |}];
  let t2 = Tensor.convert_rc_tensor_to_gc t in
  let t3 = Tensor.convert_rc_tensor_to_gc t2 in
  print_s [%message (Tensor.For_testing.get_refcount t : int)];
  Gc.keep_alive t2;
  [%expect {| ("Tensor.For_testing.get_refcount t" 3) |}];
  Gc.full_major ();
  Gc.keep_alive t3;
  print_s [%message (Tensor.For_testing.get_refcount t : int)];
  [%expect {| ("Tensor.For_testing.get_refcount t" 2) |}];
  Gc.full_major ();
  print_s [%message (Tensor.For_testing.get_refcount t : int)];
  [%expect {| ("Tensor.For_testing.get_refcount t" 1) |}]
;;

let%expect_test "with_scope handles exceptions correctly" =
  (* Unsafely grab a reference to a tensor created in the inner scope *)
  let tref = ref None in
  (try
     Tensor.with_rc_scope (fun () ->
       let t = Tensor.zeros [ 1 ] in
       tref := Some (Tensor.For_testing.globalize_gc_tensor t);
       (* Increment refcount to 2 *)
       Tensor.For_testing.increment_refcount t;
       printf "%d\n" (Tensor.For_testing.get_refcount (Option.value_exn !tref));
       [%expect {| 2 |}];
       failwith "Supposed to fail")
   with
   | Failure err -> print_endline err);
  [%expect {| Supposed to fail |}];
  let t = Option.value_exn !tref in
  printf "%d\n" (Tensor.For_testing.get_refcount t);
  (* Refcount should have been decremented to 1 *)
  [%expect {| 1 |}];
  Tensor.For_testing.decrement_refcount t
;;

let%expect_test "with_scope_tensor handles exceptions correctly" =
  (* Unsafely grab a reference to a tensor created in the inner scope *)
  let tref = ref None in
  (try
     Tensor.with_rc_scope (fun () ->
       let t =
         Tensor.with_rc_scope_tensor (fun () -> exclave_
           let t = Tensor.zeros [ 1 ] in
           (* Increment refcount to 2 *)
           Tensor.For_testing.increment_refcount t;
           printf "%d\n" (Tensor.For_testing.get_refcount t);
           [%expect {| 2 |}];
           tref := Some (Tensor.For_testing.globalize_gc_tensor t);
           ignore (failwith "Supposed to fail" : unit);
           t)
       in
       [%expect.unreachable];
       Tensor.print t [@nontail])
   with
   | Failure err -> print_endline err);
  [%expect {| Supposed to fail |}];
  let t = Option.value_exn !tref in
  printf "%d\n" (Tensor.For_testing.get_refcount t);
  [%expect {| 1 |}];
  (* Refcount should have been decremented to 1 *)
  Tensor.For_testing.decrement_refcount t
;;

let%expect_test "with_scope_tensors handles exceptions correctly" =
  let tref = ref [] in
  (try
     Tensor.with_rc_scope (fun () ->
       let tensors =
         Tensor.with_rc_scope_tensors (fun () -> exclave_
           let t1 = Tensor.zeros [ 1 ] in
           let t2 = Tensor.zeros [ 2 ] in
           let t3 = Tensor.zeros [ 3 ] in
           let tensors = [ t1; t2; t3 ] in
           Torch_local_iterators.List.iter_local
             tensors
             ~f:Tensor.For_testing.increment_refcount;
           Torch_local_iterators.List.iter_local tensors ~f:(fun t ->
             printf "%d " (Tensor.For_testing.get_refcount t));
           [%expect {| 2 2 2 |}];
           tref
           := Torch_local_iterators.List.map_local_input
                tensors
                ~f:Tensor.For_testing.globalize_gc_tensor;
           ignore (failwith "Supposed to fail" : unit);
           tensors)
       in
       [%expect.unreachable];
       Torch_local_iterators.List.iter_local tensors ~f:(fun t ->
         printf "%d " (Tensor.For_testing.get_refcount t))
       [@nontail])
   with
   | Failure err -> print_endline err);
  [%expect {| Supposed to fail |}];
  let tensors = !tref in
  Torch_local_iterators.List.iter_local tensors ~f:(fun t ->
    printf "%d " (Tensor.For_testing.get_refcount t));
  (* Refcount should have been decremented to 1 *)
  [%expect {| 1 1 1 |}];
  Torch_local_iterators.List.iter_local tensors ~f:Tensor.For_testing.decrement_refcount
;;

module Ctypes = struct
  include Ctypes_flat.Ctypes [@@alert "-deprecated"]
end

let%expect_test "Stack allocated gc tensor is globalized properly" =
  let f x =
    let managed = x in
    let fatptr =
      Ctypes_ptr.Fat.make
        ~managed:(Some (Obj.repr managed))
        ~reftyp:Ctypes.void
        (managed : nativeint)
    in
    let t = stack_ (CPointer fatptr : Tensor.t) in
    (* If we make this [Obj.magic Obj.magic] then the test fails *)
    let t_escape = Tensor.For_testing.globalize_gc_tensor t in
    Stdio.printf "inside call %nd\n" (Ctypes.raw_address_of_ptr t_escape);
    t_escape
  in
  let t1 = f 100n in
  [%expect {| inside call 100 |}];
  Stdio.printf "returned t1 %nd\n" (Ctypes.raw_address_of_ptr t1);
  [%expect {| returned t1 100 |}];
  let t = f 999n in
  [%expect {| inside call 999 |}];
  Stdio.printf "t1 %nd\n" (Ctypes.raw_address_of_ptr t1);
  [%expect {| t1 100 |}];
  Stdio.printf "t2 %nd\n" (Ctypes.raw_address_of_ptr t);
  [%expect {| t2 999 |}]
;;

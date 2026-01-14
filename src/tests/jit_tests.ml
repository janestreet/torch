open Base
open Torch

let%expect_test _ =
  let model = Module.load "foo.pt" in
  let output = Module.forward model [ Tensor.f 42.; Tensor.f 1337. ] in
  Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn output);
  [%expect {| 1463 |}]
;;

let%expect_test _ =
  let model = Module.load "foo2.pt" in
  let ivalue = Module.forward_ model [ Tensor (Tensor.f 42.); Tensor (Tensor.f 1337.) ] in
  Stdlib.Gc.full_major ();
  let t1, t2 =
    match ivalue with
    | Tuple [ Tensor t1; Tensor t2 ] -> t1, t2
    | _ -> assert false
  in
  Stdio.printf
    !"%{sexp:float} %{sexp:float}\n"
    (Tensor.to_float0_exn t1)
    (Tensor.to_float0_exn t2);
  [%expect {| 1421 -1295 |}]
;;

let%expect_test _ =
  let model = Module.load "foo3.pt" in
  let output = Module.forward model [ Tensor.of_float1 [| 1.0; 2.0; 3.0; 4.0; 5.0 |] ] in
  Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn output);
  [%expect {| 120 |}]
;;

(* test module generated with these Python code:

   {v
   import torch
   @torch.jit.script
   def would_raise(x):
   raise RuntimeError("Raising expcetion on purpose")
   return x
   torch.jit.save(would_raise, '/tmp/raise.pt')
   v}
*)
let%expect_test "test exception raise in torch script can be properly caught in OCaml" =
  let model = Module.load "raise.pt" in
  try
    let output = Module.forward model [ Tensor.of_float0 0. ] in
    ignore output
  with
  | Failure failure ->
    Stdio.print_s [%message "Exception raised and caught" (failure : string)];
    ();
    [%expect
      {|
      ("Exception raised and caught"
       (failure
         "The following operation failed in the TorchScript interpreter.\
        \nTraceback of TorchScript, serialized code (most recent call last):\
        \n  File \"code/__torch__.py\", line 8, in forward\
        \n    x: Tensor) -> NoneType:\
        \n    _0 = uninitialized(NoneType)\
        \n    ops.prim.RaiseException(\"Raising expcetion on purpose\", \"builtins.RuntimeError\")\
        \n    ~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE\
        \n    return _0\
        \n\
        \nTraceback of TorchScript, original code (most recent call last):\
        \n  File \"/tmp/ipykernel_741402/1182469162.py\", line 5, in forward\
        \n@torch.jit.script\
        \ndef would_raise(x):\
        \n    raise RuntimeError(\"Raising expcetion on purpose\")\
        \n    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE\
        \n    return x\
        \nbuiltins.RuntimeError: Raising expcetion on purpose\
        \n"))
      |}]
;;

let%expect_test _ =
  (* test that we can list all the buffers in a module, modify them, and get different
     results *)
  (* This model just adds all the buffers and parameters together. *)
  let open Base in
  let model = Module.load "w_buffers.pt" in
  let output = Module.forward model [] in
  Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn output);
  [%expect {| 10 |}];
  let buffers = Module.named_buffers model in
  Map.iter_keys ~f:Stdio.print_endline buffers;
  [%expect
    {|
    buffer0
    buffer1
    |}];
  let buffer0 = Map.find_exn buffers "buffer0" in
  let (_ : Tensor.t) = Tensor.add_ buffer0 (Tensor.ones []) in
  let output = Module.forward model [] in
  Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn output);
  [%expect {| 11 |}]
;;

let%expect_test _ =
  let (ivalues : Ivalue.t list) =
    [ None
    ; Int 42
    ; Double 1.5
    ; Tuple []
    ; Tuple [ Tuple [ None; Int 42 ]; String "foo"; Double 1.5; String "bar" ]
    ; String "foobar"
    ]
  in
  List.iter ivalues ~f:(fun ivalue ->
    Ivalue.to_raw ivalue |> Ivalue.of_raw |> Ivalue.to_string |> Stdio.print_endline);
  [%expect
    {|
    none
    42
    1.5
    ()
    ((none, 42), "foo", 1.5, "bar")
    "foobar"
    |}]
;;

open! Core
open! Async

let%expect_test "incompatible libtorch aborts on the first Torch call" =
  let%bind () = Expect_test_helpers_async.run "./probe.exe" [] in
  [%expect {| |}];
  let%bind () =
    Expect_test_helpers_async.run
      ~extend_env:[ "LD_PRELOAD", "./bad_torch_abi.so" ]
      "./probe.exe"
      []
  in
  [%expect
    {|
    ("Unclean exit" (Signal sigabrt))
    --- STDERR ---
    ocaml-torch: FATAL: Torch version mismatch.
      This OCaml code was compiled against libtorch 2.9.0 (abi tag 0),
      but the libtorch loaded into this process is 0.0.0 (abi tag 0)
    |}];
  return ()
;;

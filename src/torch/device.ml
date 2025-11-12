open Core

module T = struct
  type t = Torch_core.Device.t =
    | Cpu
    | Cuda of int
  [@@deriving bin_io, sexp, compare, hash]

  include (val Core.Core_stable.Comparator.V1.make ~compare ~sexp_of_t)
end

include T
include Comparable.Make_binable_using_comparator (T)
include Hashable.Make_binable_and_derive_hash_fold_t (T)

let of_string device =
  match String.split device ~on:':' with
  | [ "cpu" ] -> Cpu
  | [ "cuda"; x ] -> Cuda (Int.of_string x)
  | _ -> raise_s [%message "cannot parse device" (device : string)]
;;

let cuda_if_available () = if Cuda.is_available () then Cuda 0 else Cpu

let is_cuda = function
  | Cpu -> false
  | Cuda _ -> true
;;

let get_num_threads = Torch_core.Wrapper.get_num_threads
let set_num_threads = Torch_core.Wrapper.set_num_threads

let with_num_threads num_threads ~f =
  let previous_num_threads = get_num_threads () in
  set_num_threads num_threads;
  Exn.protect ~f ~finally:(fun () -> set_num_threads previous_num_threads)
;;

let%expect_test "with_num_threads" =
  let assert_ n = assert (Int.( = ) n (get_num_threads ())) in
  let original_num = get_num_threads () in
  set_num_threads 1;
  assert_ 1;
  with_num_threads 2 ~f:(fun () ->
    assert_ 2;
    set_num_threads 3;
    with_num_threads 4 ~f:(fun () ->
      assert_ 4;
      set_num_threads 5;
      assert_ 5);
    assert_ 3);
  assert_ 1;
  (try
     with_num_threads 2 ~f:(fun () -> with_num_threads 3 ~f:(fun () -> failwith "fail"))
   with
   | _ ->
     assert_ 1;
     set_num_threads 2);
  assert_ 2;
  set_num_threads original_num
;;

let%expect_test _ =
  let devices_to_int =
    [ Cpu, 1
    ; Cuda 0, 2
    ; Cuda 1, 3
    ; Cuda 2, 4
    ; Cuda 3, 5
    ; Cuda 4, 6
    ; Cuda 5, 7
    ; Cuda 6, 8
    ; Cuda 7, 9
    ]
  in
  let device_map = Map.of_alist_exn devices_to_int in
  let device_hash_table = Table.of_alist_exn devices_to_int in
  print_s [%message (device_map : int Map.t)];
  print_s [%message (device_hash_table : int Table.t)];
  [%expect
    {|
    (device_map
     ((Cpu 1) ((Cuda 0) 2) ((Cuda 1) 3) ((Cuda 2) 4) ((Cuda 3) 5) ((Cuda 4) 6)
      ((Cuda 5) 7) ((Cuda 6) 8) ((Cuda 7) 9)))
    (device_hash_table
     ((Cpu 1) ((Cuda 0) 2) ((Cuda 1) 3) ((Cuda 2) 4) ((Cuda 3) 5) ((Cuda 4) 6)
      ((Cuda 5) 7) ((Cuda 6) 8) ((Cuda 7) 9)))
    |}];
  let devices_to_int = [ Cpu, 1; Cuda 0, 2; Cuda 0, 3 ] in
  let device_map = Map.of_alist devices_to_int in
  let device_hash_table = Table.of_alist devices_to_int in
  print_s [%message (device_map : [ `Duplicate_key of t | `Ok of int Map.t ])];
  print_s [%message (device_hash_table : [ `Duplicate_key of t | `Ok of int Table.t ])];
  [%expect
    {|
    (device_map (Duplicate_key (Cuda 0)))
    (device_hash_table (Duplicate_key (Cuda 0)))
    |}];
  let devices_to_int = [ Cpu, 1; Cpu, 2 ] in
  let device_map = Map.of_alist devices_to_int in
  let device_hash_table = Table.of_alist devices_to_int in
  print_s [%message (device_map : [ `Duplicate_key of t | `Ok of int Map.t ])];
  print_s [%message (device_hash_table : [ `Duplicate_key of t | `Ok of int Table.t ])];
  [%expect
    {|
    (device_map (Duplicate_key Cpu))
    (device_hash_table (Duplicate_key Cpu))
    |}]
;;

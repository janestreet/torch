open Core

let iter_local (list @ local) ~(f : 'a @ local -> unit) =
  List.fold__local__global list ~init:() ~f:(fun () x -> f x)
;;

let rec iter2_local_exn (a @ local) (b @ local) ~f =
  match a, b with
  | [], [] -> ()
  | a_hd :: a_tl, b_hd :: b_tl ->
    f a_hd b_hd;
    iter2_local_exn a_tl b_tl ~f
  | _ -> raise_s [%message "Unequal lengths in iter2"]
;;

let map_local_input (list @ local) ~(f : 'a @ local -> 'b) =
  List.fold_right__local__global list ~init:[] ~f:(fun x prev_list -> f x :: prev_list)
;;

let hd_exn (list @ local) =
  match list with
  | [] -> raise_s [%message "Empty list"]
  | x :: _ -> x
;;

let init_local num_items ~(f : (int -> 'a @ local) @ local) : 'a List.t @ local =
  let idxs = List.range 0 num_items in
  exclave_
  List.fold_right__global__local idxs ~init:[] ~f:(fun idx prev_list -> exclave_
    f idx :: prev_list)
;;

let unzip_local (l : ('a * 'b) list @ local) : ('a list * 'b list) @ local = exclave_
  List.fold_right__local__local
    l
    ~init:([], [])
    ~f:(fun (a, b) (a_list, b_list) -> exclave_ a :: a_list, b :: b_list)
;;

let%expect_test "iter_local" =
  let nums = List.range 0 3 in
  iter_local nums ~f:(fun x -> print_endline (x |> Int.globalize |> Int.to_string));
  [%expect
    {|
    0
    1
    2
    |}]
;;

let%expect_test "iter2_local_exn" =
  let a = List.range 0 3 in
  let b = List.range 3 6 in
  iter2_local_exn a b ~f:(fun a_elem b_elem ->
    let a_elem = Int.globalize a_elem in
    let b_elem = Int.globalize b_elem in
    print_s [%message "" ~a_elem:(a_elem : int) ~b_elem:(b_elem : int)]);
  [%expect
    {|
    ((a_elem 0) (b_elem 3))
    ((a_elem 1) (b_elem 4))
    ((a_elem 2) (b_elem 5))
    |}]
;;

let%expect_test "map_local_input" =
  let nums = List.range 0 3 in
  iter_local nums ~f:(fun x -> print_endline (x |> Int.globalize |> Int.to_string));
  [%expect
    {|
    0
    1
    2
    |}];
  let f x = x + 3 in
  let new_nums = map_local_input nums ~f in
  iter_local new_nums ~f:(fun x -> print_endline (x |> Int.globalize |> Int.to_string));
  [%expect
    {|
    3
    4
    5
    |}]
;;

let%expect_test "hd_exn" =
  let nums = init_local 3 ~f:Fn.id in
  iter_local nums ~f:(fun x -> print_endline (x |> Int.globalize |> Int.to_string));
  [%expect
    {|
    0
    1
    2
    |}];
  let num = hd_exn nums in
  print_endline (num |> Int.globalize |> Int.to_string);
  [%expect {| 0 |}]
;;

let%expect_test "init_local" =
  let nums = init_local 3 ~f:(fun x -> x + 1) in
  iter_local nums ~f:(fun x -> print_endline (x |> Int.globalize |> Int.to_string));
  [%expect
    {|
    1
    2
    3
    |}]
;;

let%expect_test "unzip_local" =
  let nums = init_local 3 ~f:(fun x -> x, -x) in
  let a, b = unzip_local nums in
  iter_local a ~f:(fun x -> print_endline (x |> Int.globalize |> Int.to_string));
  [%expect
    {|
    0
    1
    2
    |}];
  iter_local b ~f:(fun x -> print_endline (x |> Int.globalize |> Int.to_string));
  [%expect
    {|
    0
    -1
    -2
    |}]
;;

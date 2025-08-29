open Core
open Torch_refcounted_bindings.Type_defs

external increment_refcount_c : nativeint -> unit = "increment_refcount"
external decrement_refcount_c : nativeint -> unit = "decrement_refcount"
external get_refcount_c : nativeint -> int = "get_refcount"

let globalize_gc_tensor = Wrapper_utils.globalize_gc_tensor

let increment_refcount (t : gc_tensor) =
  let raw_addr = Ctypes.raw_address_of_ptr (globalize_gc_tensor t) in
  increment_refcount_c raw_addr
;;

let decrement_refcount (t : gc_tensor) =
  let raw_addr = Ctypes.raw_address_of_ptr (globalize_gc_tensor t) in
  decrement_refcount_c raw_addr
;;

let get_refcount (t : gc_tensor) =
  let raw_addr = Ctypes.raw_address_of_ptr (globalize_gc_tensor t) in
  get_refcount_c raw_addr
;;

let convert_rc_tensor_to_gc (t : gc_tensor) =
  increment_refcount t;
  let t = globalize_gc_tensor t in
  Gc.Expert.add_finalizer_exn t decrement_refcount;
  t
;;

module Tensor_scope = struct
  type t = { mutable items : gc_tensor list }

  let create () = { items = [] }
  let add t item = t.items <- item :: t.items

  let clean_up t =
    List.iter t.items ~f:decrement_refcount;
    t.items <- []
  ;;
end

let scope_stack : Tensor_scope.t Stack.t = Stack.create ()

let get_current_scope_exn () =
  match Stack.top scope_stack with
  | Some scope -> scope
  | None ->
    raise_s
      [%message
        "ocaml-torch: Tried to access the current scope but the scope stack is empty, \
         add a [Tensor.with_rc_scope] around the tensor-related code"]
;;

let add_to_current_scope_exn tensor = Tensor_scope.add (get_current_scope_exn ()) tensor

let set_up_new_scope () =
  let inner_scope = Tensor_scope.create () in
  Stack.push scope_stack inner_scope
;;

let list_iter_local ~f list = List.fold__local__local list ~init:() ~f:(fun () x -> f x)

let clean_up_current_scope ~tensors_to_shift_out =
  let current_scope = get_current_scope_exn () in
  ignore (Stack.pop_exn scope_stack : Tensor_scope.t);
  if List.length tensors_to_shift_out > 0
  then (
    let outer_scope = get_current_scope_exn () in
    list_iter_local tensors_to_shift_out ~f:(fun tensor ->
      increment_refcount tensor;
      Tensor_scope.add outer_scope (globalize_gc_tensor tensor);
      ()));
  Tensor_scope.clean_up current_scope
;;

let assert_outer_scope_exists () = ignore (get_current_scope_exn () : Tensor_scope.t)

let with_rc_scope_tensor (f : unit -> gc_tensor) : gc_tensor =
  (* We have different [with_scope] functions because when users want to return
      tensor(s) from the callback, we need to ensure they are handed off to the outer
      scope.
      Tensors cannot be returned from regular [with_scope] because it returns ['a] which
      is not local. They must go through this function or the list version which will add
      them to the outer scope. *)
  assert_outer_scope_exists ();
  set_up_new_scope ();
  let returned_tensor =
    (* We don't use [exclave_] on this call because the variables inside the callback will
       be allocated on the caller's stack, which could get expensive for nested calls. *)
    try f () with
    | exn ->
      clean_up_current_scope ~tensors_to_shift_out:[];
      raise exn
  in
  clean_up_current_scope ~tensors_to_shift_out:[ returned_tensor ];
  let globalized_tensor = globalize_gc_tensor returned_tensor in
  globalized_tensor
;;

let with_rc_scope_tensors (f : unit -> gc_tensor list) : gc_tensor list =
  assert_outer_scope_exists ();
  set_up_new_scope ();
  let returned_tensors =
    try f () with
    | exn ->
      clean_up_current_scope ~tensors_to_shift_out:[];
      raise exn
  in
  clean_up_current_scope ~tensors_to_shift_out:returned_tensors;
  let globalized_tensors = List.globalize globalize_gc_tensor returned_tensors in
  globalized_tensors
;;

let with_rc_scope (f : unit -> 'a) : 'a =
  set_up_new_scope ();
  Exn.protect ~f ~finally:(fun () -> clean_up_current_scope ~tensors_to_shift_out:[])
  [@nontail]
;;

let size_to_print size_in_bytes =
  if size_in_bytes < 2 * 1024
  then size_in_bytes, "bytes"
  else if size_in_bytes < 2 * 1024 * 1024
  then size_in_bytes / 1024, "KiB"
  else if size_in_bytes < 2 * 1024 * 1024 * 1024
  then size_in_bytes / 1024 / 1024, "MiB"
  else size_in_bytes / 1024 / 1024 / 1024, "GiB"
;;

let print_rc_scopes_tensors_and_refcounts ~shape ~kind =
  let stack_depth = ref 0 in
  Stack.iter scope_stack ~f:(fun { Tensor_scope.items } ->
    if !stack_depth > 0 then print_endline "\n";
    print_endline
      [%string
        "Scope at depth %{!stack_depth#Int} with %{List.length items#Int} tensors:"];
    List.iter items ~f:(fun tensor ->
      let refcount = get_refcount tensor in
      let shape = shape tensor in
      let size = List.fold shape ~init:1 ~f:( * ) in
      let size = size * Torch_wrapper_types.Kind.size_in_bytes (kind tensor) in
      let size, unit = size_to_print size in
      let shape = [%sexp_of: int list] shape in
      print_endline
        [%string
          "shape: %{shape#Sexp}, refcount: %{refcount#Int}, size: %{size#Int} %{unit}"]);
    incr stack_depth)
;;

module For_users = struct
  let with_rc_scope = with_rc_scope
  let with_rc_scope_tensor = with_rc_scope_tensor
  let with_rc_scope_tensors = with_rc_scope_tensors
  let convert_rc_tensor_to_gc = convert_rc_tensor_to_gc
  let print_rc_scopes_tensors_and_refcounts = print_rc_scopes_tensors_and_refcounts
end

module For_testing = struct
  let increment_refcount = increment_refcount
  let decrement_refcount = decrement_refcount
  let get_refcount = get_refcount
  let globalize_gc_tensor = globalize_gc_tensor
end

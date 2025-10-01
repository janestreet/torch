open Core
open Torch_refcounted_bindings.Type_defs

external increment_refcount_c : nativeint -> unit = "increment_refcount"
external decrement_refcount_c : nativeint -> unit = "decrement_refcount"
external get_refcount_c : nativeint -> int = "get_refcount"

let globalize_gc_tensor = Wrapper_utils.globalize_gc_tensor

let increment_refcount (t : gc_tensor @ local) =
  let raw_addr = Ctypes.raw_address_of_ptr (globalize_gc_tensor t) in
  increment_refcount_c raw_addr
;;

let decrement_refcount (t : gc_tensor @ local) =
  let raw_addr = Ctypes.raw_address_of_ptr (globalize_gc_tensor t) in
  decrement_refcount_c raw_addr
;;

let get_refcount (t : gc_tensor @ local) =
  let raw_addr = Ctypes.raw_address_of_ptr (globalize_gc_tensor t) in
  get_refcount_c raw_addr
;;

(* Calling [convert_rc_tensor_to_gc] on an already gc managed tensor is fine. This works
   correctly not because there are multiple finalizers (though that would work). Instead,
   [globalize_gc_tensor] creates a new object every time so the GC doesn't know these are
   the same thing.

   Remember we also attach finalizer in [add_to_current_scope]. Consider if it makes sense
   to update that function too when changing this one. *)
let convert_rc_tensor_to_gc (t : gc_tensor @ local) =
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
let warn_on_empty_rc_scope_stack = ref false

let get_current_scope () =
  match Stack.top scope_stack with
  | Some _ as some -> some
  | None ->
    if !warn_on_empty_rc_scope_stack
    then (
      print_s
        [%message
          "ocaml-torch: Tried to access the current scope but the scope stack is empty, \
           add a [Tensor.with_rc_scope] around the tensor-related code"];
      warn_on_empty_rc_scope_stack := false);
    None
;;

let add_to_current_scope tensor =
  match get_current_scope () with
  | Some scope ->
    Tensor_scope.add scope tensor;
    tensor
  | None ->
    (* Similar to [convert_rc_tensor_to_gc], but don't increment the ref count. New
       tensors need an owner and if it's not the [Tensor_scope] (from the above branch of
       the match-statement) then the GC has to decrement the ref count from 1 to 0. *)
    Gc.Expert.add_finalizer_exn tensor decrement_refcount;
    tensor
;;

let set_up_new_scope () =
  let inner_scope = Tensor_scope.create () in
  Stack.push scope_stack inner_scope
;;

let pop_current_scope () = Stack.pop_exn scope_stack |> Tensor_scope.clean_up

(** Same as [pop_current_scope] but transfers the given tensors to the parent scope *)
let pop_current_scope_and_transfer ~tensors_to_shift_out =
  let current_scope = Stack.pop_exn scope_stack in
  let tensors =
    match get_current_scope () with
    | Some outer_scope ->
      List.globalize
        (fun tensor ->
          increment_refcount tensor;
          let tensor = globalize_gc_tensor tensor in
          Tensor_scope.add outer_scope tensor;
          tensor)
        tensors_to_shift_out
    | None -> List.globalize convert_rc_tensor_to_gc tensors_to_shift_out
  in
  Tensor_scope.clean_up current_scope;
  tensors
;;

let with_rc_scope_tensor (f : (unit -> gc_tensor @ local) @ local) : gc_tensor @ local =
  (* We have different [with_scope] functions because when users want to return
      tensor(s) from the callback, we need to ensure they are handed off to the outer
      scope.
      Tensors cannot be returned from regular [with_scope] because it returns ['a] which
      is not local. They must go through this function or the list version which will add
      them to the outer scope. *)
  set_up_new_scope ();
  let returned_tensor =
    (* We don't use [exclave_] on this call because the variables inside the callback will
       be allocated on the caller's stack, which could get expensive for nested calls. *)
    try f () with
    | exn ->
      pop_current_scope ();
      raise exn
  in
  pop_current_scope_and_transfer ~tensors_to_shift_out:[ returned_tensor ] |> List.hd_exn
;;

let with_rc_scope_tensors (f : (unit -> gc_tensor list @ local) @ local)
  : gc_tensor list @ local
  =
  set_up_new_scope ();
  let returned_tensors =
    try f () with
    | exn ->
      pop_current_scope ();
      raise exn
  in
  pop_current_scope_and_transfer ~tensors_to_shift_out:returned_tensors [@nontail]
;;

let with_rc_scope (f : (unit -> 'a) @ local) : 'a =
  set_up_new_scope ();
  Exn.protect ~f ~finally:pop_current_scope [@nontail]
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
  let warn_on_empty_rc_scope_stack = warn_on_empty_rc_scope_stack
end

module For_testing = struct
  let increment_refcount = increment_refcount
  let decrement_refcount = decrement_refcount
  let get_refcount = get_refcount
  let globalize_gc_tensor = globalize_gc_tensor
end

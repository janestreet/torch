open Core
open Torch_refcounted_bindings.Type_defs

external increment_refcount_c : tensor -> unit = "increment_refcount"
external decrement_refcount_c : tensor -> unit = "decrement_refcount"
external get_refcount_c : tensor -> int = "get_refcount"

let increment_refcount (t : tensor @ local) =
  let raw_addr = globalize_tensor t in
  increment_refcount_c raw_addr
;;

let decrement_refcount (t : tensor @ local) =
  let raw_addr = globalize_tensor t in
  decrement_refcount_c raw_addr
;;

let get_refcount (t : tensor @ local) =
  let raw_addr = globalize_tensor t in
  get_refcount_c raw_addr
;;

module Expert = struct
  let add_unmanaged_reference (t : tensor @ local) =
    increment_refcount t;
    globalize_tensor t
  ;;

  let remove_unmanaged_reference (t : tensor) = decrement_refcount t
end

(* Calling [convert_rc_tensor_to_gc] on an already gc managed tensor is fine.
   [globalize_tensor] creates a new object even if its input is already global, but even
   if it didn't the number of increments and finalizers would match.

   Remember we also attach finalizer in [add_to_current_scope]. Consider if it makes sense
   to update that function too when changing this one. *)
let convert_rc_tensor_to_gc (t : tensor @ local) =
  let t = Expert.add_unmanaged_reference t in
  Gc.Expert.add_finalizer_exn t Expert.remove_unmanaged_reference;
  t
;;

module Tensor_scope : sig
  type t
  type tensor_scope := t

  val add : t -> tensor -> unit

  module Debug : sig
    val items : t -> tensor Vec.t
  end

  module Pool : sig
    type t

    val create : unit -> t
    val alloc : t -> tensor_scope
    val clean_up : t -> tensor_scope -> unit
  end
end = struct
  type t = tensor Vec.t

  let add = Vec.push_back

  module Debug = struct
    let items = Fn.id
  end

  module Pool = struct
    type nonrec t = t Vec.t

    let create () = Vec.create ()

    let alloc (t : t) =
      match Vec.pop_back t with
      | This scope -> scope
      | Null -> Vec.create ()
    ;;

    let clean_up (t : t) scope =
      Vec.iter scope ~f:decrement_refcount;
      Vec.clear scope;
      Vec.push_back t scope
    ;;
  end
end

let global_scope_pool = Tensor_scope.Pool.create ()
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
  let inner_scope = Tensor_scope.Pool.alloc global_scope_pool in
  Stack.push scope_stack inner_scope
;;

let pop_current_scope () =
  let scope = Stack.pop_exn scope_stack in
  Tensor_scope.Pool.clean_up global_scope_pool scope
;;

(** Same as [pop_current_scope] but transfers the given tensors to the parent scope *)
let pop_current_scope_and_transfer ~tensors_to_shift_out =
  let current_scope = Stack.pop_exn scope_stack in
  let tensors =
    match get_current_scope () with
    | Some outer_scope ->
      (List.map [@mode local])
        ~f:(fun tensor ->
          increment_refcount tensor;
          let tensor = globalize_tensor tensor in
          Tensor_scope.add outer_scope tensor;
          tensor)
        tensors_to_shift_out
    | None -> (List.map [@mode local]) ~f:convert_rc_tensor_to_gc tensors_to_shift_out
  in
  Tensor_scope.Pool.clean_up global_scope_pool current_scope;
  tensors
;;

let with_rc_scope_tensor (f : (unit -> tensor @ local) @ local) : tensor @ local =
  (* We have different [with_scope] functions because when users want to return tensor(s)
     from the callback, we need to ensure they are handed off to the outer scope. Tensors
     cannot be returned from regular [with_scope] because it returns ['a] which is not
     local. They must go through this function or the list version which will add them to
     the outer scope. *)
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

let with_rc_scope_tensors (f : (unit -> tensor list @ local) @ local)
  : tensor list @ local
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
  Stack.iter scope_stack ~f:(fun scope ->
    let items = Tensor_scope.Debug.items scope in
    if !stack_depth > 0 then print_endline "\n";
    print_endline
      [%string "Scope at depth %{!stack_depth#Int} with %{Vec.length items#Int} tensors:"];
    Vec.iter items ~f:(fun tensor ->
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

  module Expert = Expert
end

module For_testing = struct
  let increment_refcount = increment_refcount
  let decrement_refcount = decrement_refcount
  let get_refcount = get_refcount
end

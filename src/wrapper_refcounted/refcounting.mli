open Torch_refcounted_bindings.Type_defs

val add_to_current_scope_exn : gc_tensor -> unit
val increment_refcount : gc_tensor -> unit
val decrement_refcount : gc_tensor -> unit
val get_refcount : gc_tensor -> int

(** The rules for refcounted (RC) tensors are this:
    1. Tensors can only be created inside one of the [with_rc_scope] functions. If no
       rc-scope was created, you will get a runtime error.
    2. Tensors are added to the scope they were created in. If you want a tensor to live
       longer, either return it by using [with_rc_scope_tensor] or make it be
       garbage-collected by calling [convert_rc_tensor_to_gc]. The latter is useful for
       long-lived tensors that are part of your model. [Var_store] will call this
       function. But we expect that 99% of tensors will be short-lived.
    3. Refcounted tensors are only cleaned up when their scope ends, so if the scope is
       too broad, the tensors won't be cleaned up. As a rule of thumb every loop should
       have a rc-scope inside of it so that temporary tensors are cleaned up at the end of
       each loop iteration. *)
module For_users : sig
  (** All tensors that are created while the callback runs will be added to a refcount
      (RC) scope, and when the callback ends, the reference counts of those tensors will
      be decremented, freeing them as long as they are not referenced elsewhere.

      Cannot return tensors from this callback because the return type is intentionally
      not local. To return tensors, you should use [with_rc_scope_tensor] or
      [with_rc_scope_tensors].

      This can be nested with any combination of the other functions. *)
  val with_rc_scope : (unit -> 'a) -> 'a

  (** Same as [with_rc_scope], but the callback should return a tensor, and that tensor
      will be safely added to the scope of the caller of this function. *)
  val with_rc_scope_tensor : (unit -> gc_tensor) -> gc_tensor

  (** Same as [with_rc_scope_tensor], but for a list of tensors. *)
  val with_rc_scope_tensors : (unit -> gc_tensor list) -> gc_tensor list

  (** Turn a refcounted tensor into a garbage collected tensor. After this function, the
      tensor can be passed around freely, but it will only be cleaned up when the
      finalizer runs. This is useful for long-lived tensors. *)
  val convert_rc_tensor_to_gc : gc_tensor -> gc_tensor

  (** Debugging function for understanding memory consumption. Iterates the current stack
      of scopes and prints all tensors that are currently allocated. *)
  val print_rc_scopes_tensors_and_refcounts
    :  shape:(gc_tensor -> int list)
    -> kind:(gc_tensor -> Torch_wrapper_types.Kind.packed)
    -> unit
end

module For_testing : sig
  val increment_refcount : gc_tensor -> unit
  val decrement_refcount : gc_tensor -> unit
  val get_refcount : gc_tensor -> int
  val globalize_gc_tensor : gc_tensor -> gc_tensor
end

# Reference counted tensors tutorial
# How to use reference counted tensors

To manage tensor memory, we introduce scopes. You can create a scope, do some work inside
it, and all tensors created while you are in the scope will be cleaned up at the end. For
example:

```
let%expect_test "addition" =
  Tensor.with_rc_scope (fun () ->
    let t = Tensor.(f 41. + f 1.) in
    Stdio.printf !"%{sexp:float}\n" (Tensor.to_float0_exn t);
    [%expect {| 42 |}];
    let t = Tensor.float_vec [ 1.; 42.; 1337. ] in
    Stdio.printf !"%{sexp:float array}\n" Tensor.(to_float1_exn (t * t));
    [%expect {| (1 1764 1787569) |}];
    Stdio.printf !"%{sexp:float array}\n" Tensor.(to_float1_exn (t + f 1.5));
    [%expect {| (2.5 43.5 1338.5) |}])
;;
```

When the callback passed in to `with_rc_scope` ends, all the tensors created during the
callback will be cleaned up and the underlying memory used by `libtorch` will be available
immediately, rather than only becoming available after the garbage collector runs.

All tensors returned from the API have the `local` mode, so they cannot escape the region
they are created in. This guarantees that when the scope ends and they are cleaned up, we
won’t use them after freeing them.

When tensors are created, they are added to the innermost scope. Behind the scenes there
is a stack of scopes (that is not threadsafe yet), and new scopes get pushed on the top.
Calling tensor functions without being in a scope will cause an error.

`with_rc_scope` accepts a callback that returns a `'a`:

```
val with_rc_scope : (unit -> 'a) @ local -> 'a
```

So you can’t return a tensor or any local value, but you can return other things. This is
a safety guarantee, because returning tensors requires a handoff to the outer scope.

## Nested scopes and returning tensors
If I want to do some computation that creates intermediate tensors, and just want to keep
the final output tensor, I can return a tensor using `with_rc_scope_tensor`:

```
let%expect_test "Tensor returned from inner scope to outer" =
  Tensor.with_rc_scope_tensor (fun () ->
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
```

This will transfer the ownership of the tensor returned from the callback to the outer
scope.

The return type of the callback that is passed in should be `Tensor.t @ local`, and there
is a list version: `with_rc_scope_tensors`. There isn’t a good way to return tensors in
arbitrary data structures.

When returning locals, you will want to use `exclave_` to mark the end of the function’s
region (like in the callback passed into `with_rc_scope_tensor` above).

## Var stores for model parameters
When constructing a model, we use a var store to track model parameters. Model parameters
in the var store are not local and are garbage collected, so they can be loaded and saved
to checkpoints and passed around in arbitrary data structures.

See the [refcounted llama2 example](./examples/refcounted/llama2/README.md) for how this
works.

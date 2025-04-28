(** Runs pre-compiled AOT Inductor models.

    As of writing, only CUDA support is implemented, so this will fail when CPU is
    requested. *)

type t

(** Load an AOT inductor-compiled model from a shared object [file].

    @param so_path the shared object file containing the AOT compiled model
    @param cubin_dir
      load the cubin files from this directory instead of the directory hardcoded in
      [file]
    @param device load the model onto this device
    @param max_concurrent_executions
      maximum number of concurrent invocations of the model that are possible (default: 1) *)
val load
  :  ?device:Device.t
  -> ?max_concurrent_executions:int
  -> cubin_dir:string
  -> so_path:string
  -> unit
  -> t

(** This runs the model and discards any returned tensors, regardless of how many there
    are. *)
val run_unit : t -> Tensor.t list -> unit

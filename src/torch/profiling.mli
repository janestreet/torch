(** We expose analogous calls to
    (https://docs.pytorch.org/docs/stable/torch_cuda_memory.html), will only work when
    running on CUDA *)

val record_memory_history : unit -> unit
val save_memory_snapshot_pickled : output_filename:string -> unit

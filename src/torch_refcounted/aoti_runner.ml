open Torch_refcounted_core.Wrapper

type t = Aoti_runner_cuda.t

let load ?device ?max_concurrent_executions ~cubin_dir ~so_path () : t =
  match (device : Device.t option) with
  | Some (Cuda _ as device) ->
    Aoti_runner_cuda.load ~device ?max_concurrent_executions ~cubin_dir ~so_path ()
  | _ -> failwith "AOTI runner is not yet supported on CPU"
;;

let run_unit = Aoti_runner_cuda.run_unit

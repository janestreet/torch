open Ctypes
module Type_defs = Type_defs

module C (F : Cstubs.FOREIGN) = struct
  open Type_defs
  open F

  let manual_seed = foreign "at_manual_seed" (int64_t @-> returning void)
  let free = foreign "free" (ptr void @-> returning void)
  let get_num_threads = foreign "at_get_num_threads" (void @-> returning int)
  let set_num_threads = foreign "at_set_num_threads" (int @-> returning void)

  module Tensor = struct
    let new_tensor = foreign "at_new_tensor" (void @-> returning raw_tensor)

    let tensor_of_data =
      foreign
        "at_tensor_of_data"
        (ptr void
         (* data *)
         @-> ptr int64_t
         (* dims *)
         @-> int
         (* ndims *)
         @-> int
         (* element size in bytes *)
         @-> int
         (* kind *)
         @-> returning raw_tensor)
    ;;

    let copy_to_elements =
      foreign
        "at_copy_to_elements"
        (gc_tensor
         (* tensor *)
         @-> ptr void
         (* data *)
         @-> int64_t
         (* numel *)
         @-> int
         (* element size in bytes *)
         @-> returning void)
    ;;

    let copy_to_bytes =
      foreign
        "at_copy_to_bytes"
        (gc_tensor
         (* tensor *)
         @-> ptr void
         (* data *)
         @-> int64_t
         (* max_size *)
         @-> returning void)
    ;;

    let copy_ =
      foreign
        "at_copy_"
        (gc_tensor
         (* dst *)
         @-> gc_tensor
         (* src *)
         @-> bool
         (* non_blocking *)
         @-> returning void)
    ;;

    let set_data =
      foreign
        "at_set_data"
        (gc_tensor (* dst *) @-> gc_tensor (* src *) @-> returning void)
    ;;

    let float_vec =
      foreign
        "at_float_vec"
        (ptr double (* values *)
         @-> int (* num values *)
         @-> int (* kind *)
         @-> returning raw_tensor)
    ;;

    let int_vec =
      foreign
        "at_int_vec"
        (ptr int64_t
         (* values *)
         @-> int
         (* num values *)
         @-> int
         (* kind *)
         @-> returning raw_tensor)
    ;;

    let device = foreign "at_device" (gc_tensor @-> returning int)
    let defined = foreign "at_defined" (gc_tensor @-> returning bool)
    let num_dims = foreign "at_dim" (gc_tensor @-> returning int)
    let shape = foreign "at_shape" (gc_tensor @-> ptr int (* dims *) @-> returning void)
    let scalar_type = foreign "at_scalar_type" (gc_tensor @-> returning int)
    let use_count = foreign "at_use_count" (gc_tensor @-> returning int)
    let backward = foreign "at_backward" (gc_tensor @-> int @-> int @-> returning void)
    let requires_grad = foreign "at_requires_grad" (gc_tensor @-> returning int)
    let grad_set_enabled = foreign "at_grad_set_enabled" (int @-> returning int)
    let get = foreign "at_get" (gc_tensor @-> int @-> returning raw_tensor)

    let double_value =
      foreign
        "at_double_value_at_indexes"
        (gc_tensor @-> ptr int @-> int @-> returning double)
    ;;

    let int64_value =
      foreign
        "at_int64_value_at_indexes"
        (gc_tensor @-> ptr int @-> int @-> returning int64_t)
    ;;

    let double_value_set =
      foreign
        "at_set_double_value_at_indexes"
        (gc_tensor @-> ptr int @-> int @-> double @-> returning void)
    ;;

    let int64_value_set =
      foreign
        "at_set_int64_value_at_indexes"
        (gc_tensor @-> ptr int @-> int @-> int64_t @-> returning void)
    ;;

    let fill_double = foreign "at_fill_double" (gc_tensor @-> double @-> returning void)
    let fill_int64 = foreign "at_fill_int64" (gc_tensor @-> int64_t @-> returning void)
    let print = foreign "at_print" (gc_tensor @-> returning void)
    let to_string = foreign "at_to_string" (gc_tensor @-> int @-> returning string)

    let run_backward =
      foreign
        "at_run_backward"
        (ptr gc_tensor
         @-> int
         @-> ptr gc_tensor
         @-> int
         @-> ptr raw_tensor
         @-> int
         @-> int
         @-> returning void)
    ;;
  end

  module Scalar = struct
    let to_int64 = foreign "ats_to_int" (scalar @-> returning int64_t)
    let to_float = foreign "ats_to_float" (scalar @-> returning double)
    let int = foreign "ats_int" (int64_t @-> returning scalar)
    let float = foreign "ats_float" (float @-> returning scalar)
    let free = foreign "ats_free" (scalar @-> returning void)
  end

  module Serialize = struct
    let save = foreign "at_save" (gc_tensor @-> string @-> returning void)
    let load = foreign "at_load" (string @-> returning raw_tensor)

    let save_multi =
      foreign
        "at_save_multi"
        (ptr gc_tensor @-> ptr (ptr char) @-> int @-> string @-> returning void)
    ;;

    let load_multi =
      foreign
        "at_load_multi"
        (ptr raw_tensor @-> ptr (ptr char) @-> int @-> string @-> returning void)
    ;;

    let load_multi_ =
      foreign
        "at_load_multi_"
        (ptr gc_tensor @-> ptr (ptr char) @-> int @-> string @-> returning void)
    ;;

    let load_callback =
      foreign
        "at_load_callback"
        (string
         @-> static_funptr Ctypes.(string @-> raw_tensor @-> returning void)
         @-> returning void)
    ;;
  end

  module Optimizer = struct
    let adam =
      foreign
        "ato_adam"
        (float @-> float @-> float @-> float @-> float @-> returning optimizer)
    ;;

    let rmsprop =
      foreign
        "ato_rmsprop"
        (float
         (* learning rate *)
         @-> float
         (* alpha *)
         @-> float
         (* eps *)
         @-> float
         (* weight decay *)
         @-> float
         (* momentum *)
         @-> int
         (* centered *)
         @-> returning optimizer)
    ;;

    let sgd =
      foreign
        "ato_sgd"
        (float
         (* learning rate *)
         @-> float
         (* momentum *)
         @-> float
         (* dampening *)
         @-> float
         (* weight decay *)
         @-> bool
         (* nesterov *)
         @-> returning optimizer)
    ;;

    let add_parameters =
      foreign "ato_add_parameters" (optimizer @-> ptr gc_tensor @-> int @-> returning void)
    ;;

    let set_learning_rate =
      foreign "ato_set_learning_rate" (optimizer @-> float @-> returning void)
    ;;

    let set_momentum = foreign "ato_set_momentum" (optimizer @-> float @-> returning void)
    let zero_grad = foreign "ato_zero_grad" (optimizer @-> returning void)
    let step = foreign "ato_step" (optimizer @-> returning void)
    let free = foreign "ato_free" (optimizer @-> returning void)
  end

  module Cuda = struct
    let device_count = foreign "atc_cuda_device_count" (void @-> returning int)
    let is_available = foreign "atc_cuda_is_available" (void @-> returning int)
    let cudnn_is_available = foreign "atc_cudnn_is_available" (void @-> returning int)
    let set_benchmark_cudnn = foreign "atc_set_benchmark_cudnn" (int @-> returning void)
  end

  module Ivalue = struct
    let to_int64 = foreign "ati_to_int" (ivalue @-> returning int64_t)
    let to_bool = foreign "ati_to_bool" (ivalue @-> returning int)
    let to_double = foreign "ati_to_double" (ivalue @-> returning double)
    let to_tensor = foreign "ati_to_tensor" (ivalue @-> returning raw_tensor)
    let tuple_length = foreign "ati_tuple_length" (ivalue @-> returning int)
    let list_length = foreign "ati_list_length" (ivalue @-> returning int)

    let to_tuple =
      foreign "ati_to_tuple" (ivalue @-> ptr ivalue @-> int @-> returning void)
    ;;

    let to_tensor_list =
      foreign "ati_to_tensor_list" (ivalue @-> ptr raw_tensor @-> int @-> returning void)
    ;;

    let to_generic_list =
      foreign "ati_to_generic_list" (ivalue @-> ptr ivalue @-> int @-> returning void)
    ;;

    let to_string = foreign "ati_to_string" (ivalue @-> returning string)
    let none = foreign "ati_none" (void @-> returning ivalue)
    let bool = foreign "ati_bool" (int @-> returning ivalue)
    let tensor = foreign "ati_tensor" (gc_tensor @-> returning ivalue)
    let int64 = foreign "ati_int" (int64_t @-> returning ivalue)
    let double = foreign "ati_double" (double @-> returning ivalue)
    let tuple = foreign "ati_tuple" (ptr ivalue @-> int @-> returning ivalue)

    let tensor_list =
      foreign "ati_tensor_list" (ptr gc_tensor @-> int @-> returning ivalue)
    ;;

    let string = foreign "ati_string" (string @-> returning ivalue)
    let tag = foreign "ati_tag" (ivalue @-> returning int)
    let free = foreign "ati_free" (ivalue @-> returning void)
  end

  module Module = struct
    let load = foreign "atm_load" (string @-> int @-> returning module_)
    let load_str = foreign "atm_load_str" (string @-> int @-> int @-> returning module_)

    let forward =
      foreign "atm_forward" (module_ @-> ptr gc_tensor @-> int @-> returning raw_tensor)
    ;;

    let forward_ =
      foreign "atm_forward_" (module_ @-> ptr ivalue @-> int @-> returning ivalue)
    ;;

    let named_buffers = foreign "atm_named_buffers" (module_ @-> returning ivalue)
    let free = foreign "atm_free" (module_ @-> returning void)
  end

  module Aoti_runner_cuda = struct
    let load =
      foreign
        "aoti_runner_cuda_load"
        (string @-> int @-> int @-> string @-> returning aoti_runner_cuda)
    ;;

    let run_unit =
      foreign
        "aoti_runner_cuda_run_unit"
        (aoti_runner_cuda @-> ptr gc_tensor @-> int @-> returning void)
    ;;

    let free = foreign "aoti_runner_cuda_free" (aoti_runner_cuda @-> returning void)
  end

  module Generated = Torch_bindings_generated.C (F)
end

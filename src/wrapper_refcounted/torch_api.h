#ifndef __TORCH_API_H__
#define __TORCH_API_H__
#include <stdint.h>
#include <stddef.h>
#include <caml/custom.h>

#ifdef __cplusplus
extern "C" {
typedef torch::Scalar *scalar;
typedef torch::optim::Optimizer *optimizer;
typedef torch::jit::script::Module *module;
typedef torch::inductor::AOTIModelContainerRunnerCuda *aoti_runner_cuda;
typedef torch::jit::IValue *ivalue;
typedef torch::TensorImpl *raw_tensor;
typedef torch::TensorImpl *gc_tensor;
#define PROTECT(x)                                                                       \
  try {                                                                                  \
    x                                                                                    \
  } catch (const exception &e) {                                                         \
    caml_failwith(strdup(e.what()));                                                     \
    __builtin_unreachable();                                                             \
  }
#else
typedef void *optimizer;
typedef void *scalar;
typedef void *module;
typedef void *aoti_runner_cuda;
typedef void *ivalue;
typedef void *raw_tensor;
typedef void *gc_tensor;
#endif

void increment_refcount_internal_noalloc(value managed_tensor);
void decrement_refcount_internal(value managed_tensor);
int get_refcount_internal_noalloc(value managed_tensor);

void at_manual_seed(int64_t);
value at_new_tensor();
value at_tensor_of_data(value vs, value dims, int element_size_in_bytes, int type);
void at_copy_to_elements(value t, value vs, int64_t numel, int element_size_in_bytes);
void at_copy_to_bytes(value t, value bytes, int64_t bytes_offset, int64_t bytes_len);
void at_copy_from_bytes(value t, value bytes, int64_t bytes_offset, int64_t bytes_len);
void at_copy_from_elements(value t, value vs, int64_t numel, int element_size_in_bytes);

value at_float_vec(value values, int type);
value at_int_vec(value values, int type);

int at_defined(value);
int at_is_sparse(value);
int at_device(value);
int at_dim(value);
value at_shape(value);
int at_scalar_type(value);
int at_use_count(value);

void at_autocast_clear_cache();
int at_autocast_decrement_nesting();
int at_autocast_increment_nesting();
int at_autocast_is_enabled();
int at_autocast_set_enabled(int b);

void at_backward(value tensor, int keep_graph, int create_graph);
int at_requires_grad(value tensor);
int at_grad_set_enabled(int b);

value at_get(value, int index);
void at_fill_double(value, double);
void at_fill_int64(value, int64_t);

double at_double_value_at_indexes(value tensor, value indexes);
int64_t at_int64_value_at_indexes(value tensor, value indexes);
void at_set_double_value_at_indexes(value tensor, value indexes, double v);
void at_set_int64_value_at_indexes(value tensor, value indexes, int64_t v);

void at_copy_(value dst, value src, int nonblocking);
void at_set_data(value dst, value src);

void at_print(value tensor);
value at_to_string(value tensor, int line_size);
void at_save(value tensor, const char *filename);
value at_load(const char *filename);

int at_get_num_threads();
void at_set_num_threads(int n_threads);

void at_save_multi(value tensors, value tensor_names, const char *filename);
value at_load_multi(value tensor_names, const char *filename);
void at_load_multi_(value tensors, value tensor_names, const char *filename);

value at_load_all(const char *filename);

value at_run_backward(value tensors, value inputs, int keep_graph, int create_graph);

value ato_adam(double learning_rate, double beta1, double beta2, double weight_decay,
               double eps);
value ato_rmsprop(double learning_rate, double alpha, double eps, double weight_decay,
                  double momentum, int centered);
value ato_sgd(double learning_rate, double momentum, double dampening,
              double weight_decay, int nesterov);
void ato_add_parameters(value optimizer, value tensors);
void ato_set_learning_rate(value optimizer, double learning_rate);
void ato_set_momentum(value optimizer, double momentum);
void ato_zero_grad(value optimizer);
void ato_step(value optimizer);
void ato_free(value optimizer);

value ats_int(int64_t);
value ats_float(double);
int64_t ats_to_int(value);
double ats_to_float(value);
void ats_free(value);

int atc_cuda_device_count();
int atc_cuda_is_available();
int atc_cudnn_is_available();
void atc_set_benchmark_cudnn(int b);

value aoti_runner_cuda_load(const char *filename, int num_concurrent_executions,
                            int device, const char *cubin_dir);
void aoti_runner_cuda_run_unit(value runner, value tensors);
void aoti_runner_cuda_free(value runner);

void torch_record_memory_history();
void torch_save_memory_snapshot_pickled(const char *);

#include "torch_refcounted_api_generated.h"

#ifdef __cplusplus
};
#endif

#endif

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

void finalize_managed_tensor_internal(value managed);
value make_managed_tensor_internal(value addr);

void at_manual_seed(int64_t);
raw_tensor at_new_tensor();
raw_tensor at_tensor_of_data(void *vs, int64_t *dims, int ndims,
                             int element_size_in_bytes, int type);
void at_copy_to_elements(gc_tensor t, void *vs, int64_t numel, int element_size_in_bytes);
void at_copy_to_bytes(gc_tensor t, void *vs, int64_t max_size);

raw_tensor at_float_vec(double *values, int value_len, int type);
raw_tensor at_int_vec(int64_t *values, int value_len, int type);

int at_defined(gc_tensor);
int at_is_sparse(gc_tensor);
int at_device(gc_tensor);
int at_dim(gc_tensor);
void at_shape(gc_tensor, int *);
void at_stride(gc_tensor, int *);
int at_scalar_type(gc_tensor);
int at_use_count(gc_tensor);

void at_autocast_clear_cache();
int at_autocast_decrement_nesting();
int at_autocast_increment_nesting();
int at_autocast_is_enabled();
int at_autocast_set_enabled(int b);

void at_backward(gc_tensor, int, int);
int at_requires_grad(gc_tensor);
int at_grad_set_enabled(int);

raw_tensor at_get(gc_tensor, int index);
void at_fill_double(gc_tensor, double);
void at_fill_int64(gc_tensor, int64_t);

double at_double_value_at_indexes(gc_tensor, int *indexes, int indexes_len);
int64_t at_int64_value_at_indexes(gc_tensor, int *indexes, int indexes_len);
void at_set_double_value_at_indexes(gc_tensor, int *indexes, int indexes_len, double v);
void at_set_int64_value_at_indexes(gc_tensor, int *indexes, int indexes_len, int64_t v);

void at_copy_(gc_tensor dst, gc_tensor src, int nonblocking);
void at_set_data(gc_tensor dst, gc_tensor src);

void at_print(gc_tensor);
char *at_to_string(gc_tensor, int line_size);
void at_save(gc_tensor, char *filename);
raw_tensor at_load(char *filename);

int at_get_num_threads();
void at_set_num_threads(int n_threads);

void at_save_multi(gc_tensor *tensors, char **tensor_names, int ntensors, char *filename);
/* [at_load_multi] takes as input an array of nullptr for [tensors]. */
void at_load_multi(raw_tensor *outputs, char **tensor_names, int ntensors,
                   char *filename);
/* [at_load_multi_] takes as input an array of allocation [tensors]. */
void at_load_multi_(gc_tensor *tensors, char **tensor_names, int ntensors,
                    char *filename);

void at_load_callback(char *filename, void (*f)(char *, raw_tensor));

void at_run_backward(gc_tensor *tensors, int ntensors, gc_tensor *inputs, int ninputs,
                     raw_tensor *outputs, int keep_graph, int create_graph);

optimizer ato_adam(double learning_rate, double beta1, double beta2, double weight_decay,
                   double eps);
optimizer ato_rmsprop(double learning_rate, double alpha, double eps, double weight_decay,
                      double momentum, int centered);
optimizer ato_sgd(double learning_rate, double momentum, double dampening,
                  double weight_decay, int nesterov);
void ato_add_parameters(optimizer, gc_tensor *, int ntensors);
void ato_set_learning_rate(optimizer, double learning_rate);
void ato_set_momentum(optimizer, double momentum);
void ato_zero_grad(optimizer);
void ato_step(optimizer);
void ato_free(optimizer);

scalar ats_int(int64_t);
scalar ats_float(double);
int64_t ats_to_int(scalar);
double ats_to_float(scalar);
void ats_free(scalar);

int atc_cuda_device_count();
int atc_cuda_is_available();
int atc_cudnn_is_available();
void atc_set_benchmark_cudnn(int b);

module atm_load(char *, int);
module atm_load_str(char *, size_t, int);
raw_tensor atm_forward(module, gc_tensor *tensors, int ntensors);
ivalue atm_forward_(module, ivalue *ivalues, int nivalues);
ivalue atm_named_buffers(module);
void atm_free(module);

aoti_runner_cuda aoti_runner_cuda_load(char *filename, int num_concurrent_executions,
                                       int device, char *cubin_dir);
void aoti_runner_cuda_run_unit(aoti_runner_cuda, gc_tensor *tensors, int ntensors);
void aoti_runner_cuda_free(aoti_runner_cuda);

ivalue ati_none();
ivalue ati_tensor(gc_tensor);
ivalue ati_bool(int);
ivalue ati_int(int64_t);
ivalue ati_double(double);
ivalue ati_tuple(ivalue *, int);
ivalue ati_string(char *);
ivalue ati_generic_list(ivalue *, int);
ivalue ati_generic_dict(ivalue *, int);
ivalue ati_int_list(int64_t *, int);
ivalue ati_double_list(double *, int);
ivalue ati_bool_list(char *, int);
ivalue ati_string_list(char **, int);
ivalue ati_tensor_list(gc_tensor *, int);

raw_tensor ati_to_tensor(ivalue);
int64_t ati_to_int(ivalue);
double ati_to_double(ivalue);
char *ati_to_string(ivalue);
int ati_to_bool(ivalue);
int ati_length(ivalue);
int ati_tuple_length(ivalue);
int ati_list_length(ivalue);
void ati_to_tuple(ivalue, ivalue *, int);
void ati_to_generic_list(ivalue, ivalue *, int);
void ati_to_generic_dict(ivalue, ivalue *, int);
void ati_to_int_list(ivalue, int64_t *, int);
void ati_to_double_list(ivalue, double *, int);
void ati_to_bool_list(ivalue, char *, int);
void ati_to_tensor_list(ivalue, raw_tensor *, int);

int ati_tag(ivalue);

void ati_free(ivalue);

#include "torch_api_generated.h"

#ifdef __cplusplus
};
#endif

#endif

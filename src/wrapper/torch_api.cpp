#include <torch/csrc/autograd/engine.h>
#include <torch/torch.h>
#include <ATen/autocast_mode.h>
#include <torch/script.h>
#include <stdexcept>
#include <vector>
#include <caml/fail.h>
#include <caml/memory.h>
#undef invalid_argument
#include "torch_api.h"
#include "ctypes_cstubs_internals.h"

using namespace std;

torch::Tensor *tensor_ptr_from_ocaml(gc_tensor t) { return (torch::Tensor *)t; }

torch::Tensor tensor_from_ocaml(gc_tensor t) {
  return *tensor_ptr_from_ocaml(t);
}

CAMLprim void finalize_tensor(value block) {
  gc_tensor t = *(void **)Data_custom_val(block);
  at_free(t);
}

static struct custom_operations ops = {"torch-tensor",
                                       finalize_tensor,
                                       custom_compare_default,
                                       custom_hash_default,
                                       custom_serialize_default,
                                       custom_deserialize_default,
                                       custom_compare_ext_default,
                                       custom_fixed_length_default};

CAMLprim value with_tensor_gc_internal(raw_tensor t) {
  // See https://v2.ocaml.org/manual/intfc.html#s%3Ac-custom
  unsigned long int off_heap_cpu_memory_bytes = 0;
  torch::Tensor *tensor = (torch::Tensor *)t;
  if (tensor->defined() && tensor->device() == at::kCPU) {
    off_heap_cpu_memory_bytes = tensor->numel() * tensor->element_size();
  }
  value new_block = caml_alloc_custom_mem(&ops, sizeof(torch::Tensor *),
                                          off_heap_cpu_memory_bytes);
  *(void **)Data_custom_val(new_block) = t;
  return new_block;
}

raw_tensor tensor_to_ocaml(torch::Tensor cpp_tensor) {
  torch::Tensor *res = new torch::Tensor(cpp_tensor);
  return (raw_tensor)res;
}

void at_manual_seed(int64_t seed) { torch::manual_seed(seed); }

vector<torch::Tensor> of_carray_tensor(gc_tensor *vs, int len) {
  vector<torch::Tensor> result;
  for (int i = 0; i < len; ++i)
    result.push_back(tensor_from_ocaml(vs[i]));
  return result;
}

c10::List<c10::optional<torch::Tensor>> of_carray_tensor_opt(gc_tensor *vs,
                                                             int len) {
  vector<c10::optional<torch::Tensor>> result;
  for (int i = 0; i < len; ++i) {
    result.push_back(
        vs[i] ? c10::optional<torch::Tensor>(tensor_from_ocaml(vs[i]))
              : c10::nullopt);
  }
  return c10::List<c10::optional<torch::Tensor>>(result);
}

c10::optional<at::Device> optional_device_of_int(int d) {
  if (d == -2)
    return c10::optional<at::Device>();
  else if (d == -1)
    return c10::optional<at::Device>(at::Device(at::kCPU));
  else if (d >= 0)
    return c10::optional<at::Device>(at::Device(at::kCUDA, /*index=*/d));
  else
    throw std::invalid_argument("unknown device index");
}

at::Device device_of_int(int d) { return optional_device_of_int(d).value(); }

raw_tensor at_new_tensor() {
  PROTECT(return tensor_to_ocaml(torch::Tensor());)
  return nullptr;
}

raw_tensor at_tensor_of_data(void *vs, int64_t *dims, int ndims,
                             int element_size_in_bytes, int type) {
  PROTECT(
      torch::Tensor tensor = torch::zeros(torch::IntArrayRef(dims, ndims),
                                          torch::ScalarType(type));
      if ((int64_t)element_size_in_bytes != tensor.element_size()) throw std::
          invalid_argument("incoherent element sizes in bytes");
      void *tensor_data = tensor.data_ptr();
      memcpy(tensor_data, vs, tensor.numel() * element_size_in_bytes);
      return tensor_to_ocaml(tensor);)
  return nullptr;
}

void at_copy_data(gc_tensor t, void *vs, int64_t numel, int elt_size_in_bytes) {
  PROTECT(
      torch::Tensor *tensor = tensor_ptr_from_ocaml(t);
      if ((int64_t)elt_size_in_bytes != tensor->element_size()) throw std::
          invalid_argument("incoherent element sizes in bytes");
      if ((int64_t)numel > tensor->numel()) throw std::invalid_argument(
          "target numel is larger than tensor numel");
      if (tensor->device().type() != at::kCPU) {
        torch::Tensor tmp_tensor = tensor->to(at::kCPU).contiguous();
        void *tensor_data = tmp_tensor.data_ptr();
        memcpy(vs, tensor_data, numel * elt_size_in_bytes);
      } else {
        auto tmp_tensor = tensor->contiguous();
        void *tensor_data = tmp_tensor.data_ptr();
        memcpy(vs, tensor_data, numel * elt_size_in_bytes);
      })
}

raw_tensor at_float_vec(double *vs, int len, int type) {
  PROTECT(torch::Tensor tensor = torch::empty({len}, torch::ScalarType(type));
          for (int i = 0; i < len; ++i) tensor[i] = vs[i];
          return tensor_to_ocaml(tensor);)
  return nullptr;
}

raw_tensor at_int_vec(int64_t *vs, int len, int type) {
  PROTECT(torch::Tensor tensor = torch::empty({len}, torch::ScalarType(type));
          for (int i = 0; i < len; ++i) tensor[i] = vs[i];
          return tensor_to_ocaml(tensor);)
  return nullptr;
}

int at_defined(gc_tensor t) {
  PROTECT(return tensor_ptr_from_ocaml(t)->defined();)
  return -1;
}

int at_is_sparse(gc_tensor t) {
  PROTECT(return tensor_ptr_from_ocaml(t)->is_sparse();)
  return -1;
}

int at_dim(gc_tensor t) {
  PROTECT(return tensor_ptr_from_ocaml(t)->dim();)
  return -1;
}

void at_shape(gc_tensor t, int *dims) {
  PROTECT(int i = 0; for (int dim
                          : tensor_ptr_from_ocaml(t)->sizes()) dims[i++] = dim;)
}

void at_stride(gc_tensor t, int64_t *dims) {
  PROTECT(int i = 0;
          for (int64_t dim
               : tensor_ptr_from_ocaml(t)->strides()) dims[i++] = dim;)
}

int at_scalar_type(gc_tensor t) {
  PROTECT(return static_cast<int>(tensor_ptr_from_ocaml(t)->scalar_type());)
}

void at_autocast_clear_cache() { at::autocast::clear_cache(); }

int at_autocast_decrement_nesting() {
  PROTECT(return at::autocast::decrement_nesting();)
  return -1;
}

int at_autocast_increment_nesting() {
  PROTECT(return at::autocast::increment_nesting();)
  return -1;
}

int at_autocast_is_enabled() {
  PROTECT(return at::autocast::is_enabled();)
  return -1;
}

int at_autocast_set_enabled(int b) {
  PROTECT(bool is_enabled = at::autocast::is_enabled();
          at::autocast::set_enabled(b); return is_enabled;)
  return -1;
}

int at_device(gc_tensor t) {
  PROTECT(auto device = tensor_ptr_from_ocaml(t)->device();
          if (device.is_cpu()) return -1; return device.index();)
}

void at_backward(gc_tensor t, int keep_graph, int create_graph) {
  PROTECT(
      caml_release_runtime_system(); try {
        tensor_ptr_from_ocaml(t)->backward({}, keep_graph, create_graph);
      } catch (const exception &) {
        caml_acquire_runtime_system();
        throw;
      } caml_acquire_runtime_system();)
}

int at_requires_grad(gc_tensor t) {
  PROTECT(return tensor_ptr_from_ocaml(t)->requires_grad();)
  return -1;
}

int at_grad_set_enabled(int b) {
  PROTECT(bool is_enabled = torch::autograd::GradMode::is_enabled();
          torch::autograd::GradMode::set_enabled(b); return is_enabled;)
  return -1;
}

raw_tensor at_get(gc_tensor t, int index) {
  PROTECT(return tensor_to_ocaml(tensor_from_ocaml(t)[index]);)
  return nullptr;
}

template <typename T>
T at_value_at_indexes(gc_tensor t, int *indexes, int indexes_len) {
  PROTECT(torch::Tensor tensor = tensor_from_ocaml(t);
          for (int i = 0; i < indexes_len;
               ++i) { tensor = tensor[indexes[i]]; } return tensor.item<T>();)
  return T();
}

double at_double_value_at_indexes(gc_tensor t, int *indexes, int indexes_len) {
  return at_value_at_indexes<double>(t, indexes, indexes_len);
}

int64_t at_int64_value_at_indexes(gc_tensor t, int *indexes, int indexes_len) {
  return at_value_at_indexes<int64_t>(t, indexes, indexes_len);
}

template <typename T>
void at_set_value_at_indexes(gc_tensor t, int *indexes, int indexes_len, T v) {
  PROTECT(torch::Tensor tensor = tensor_from_ocaml(t);
          for (int i = 0; i < indexes_len;
               ++i) { tensor = tensor[indexes[i]]; } tensor.fill_(v);)
}

void at_set_double_value_at_indexes(gc_tensor t, int *indexes, int indexes_len,
                                    double v) {
  at_set_value_at_indexes<double>(t, indexes, indexes_len, v);
}

void at_set_int64_value_at_indexes(gc_tensor t, int *indexes, int indexes_len,
                                   int64_t v) {
  at_set_value_at_indexes<int64_t>(t, indexes, indexes_len, v);
}

void at_fill_double(gc_tensor t, double v) {
  PROTECT(tensor_ptr_from_ocaml(t)->fill_(v);)
}

void at_fill_int64(gc_tensor t, int64_t v) {
  PROTECT(tensor_ptr_from_ocaml(t)->fill_(v);)
}

void at_print(gc_tensor t) {
  PROTECT(torch::Tensor *tensor = (torch::Tensor *)t; cout << *tensor << endl;)
}

char *at_to_string(gc_tensor t, int line_size) {
  PROTECT(std::ostringstream oss;
          torch::print(oss, tensor_from_ocaml(t), line_size);
          return strdup(oss.str().c_str());)
  return nullptr;
}

void at_copy_(gc_tensor dst, gc_tensor src) {
  PROTECT(tensor_ptr_from_ocaml(dst)->copy_(tensor_from_ocaml(src));)
}
void at_set_data(gc_tensor dst, gc_tensor src) {
  PROTECT(tensor_ptr_from_ocaml(dst)->set_data(tensor_from_ocaml(src));)
}

void at_save(gc_tensor t, char *filename) {
  PROTECT(torch::save(tensor_from_ocaml(t), filename);)
}

void at_save_multi(gc_tensor *tensors, char **tensor_names, int ntensors,
                   char *filename) {
  PROTECT(torch::serialize::OutputArchive archive;
          for (int i = 0; i < ntensors; ++i)
              archive.write(std::string(tensor_names[i]),
                            tensor_from_ocaml(tensors[i]), /* buffer=*/false);
          archive.save_to(filename);)
}

void at_load_multi(raw_tensor *outputs, char **tensor_names, int ntensors,
                   char *filename) {
  PROTECT(torch::serialize::InputArchive archive;
          archive.load_from(std::string(filename));
          vector<torch::Tensor> ts(ntensors);
          for (int i = 0; i < ntensors; ++i)
              archive.read(std::string(tensor_names[i]), ts[i]);
          // Only allocate the new tensors now so that if there is an exception
          // raised during [read], no memory has to be freed.
          for (int i = 0; i < ntensors; ++i) outputs[i] =
              tensor_to_ocaml(ts[i]);)
}

void at_load_callback(char *filename, void (*f)(char *, raw_tensor)) {
  PROTECT(auto module = torch::jit::load(filename);
          for (const auto &p
               : module.named_parameters()) {
            auto v = p.value;
            f((char *)p.name.c_str(), tensor_to_ocaml(v));
          })
}

void at_load_multi_(gc_tensor *tensors, char **tensor_names, int ntensors,
                    char *filename) {
  PROTECT(torch::NoGradGuard no_grad; torch::serialize::InputArchive archive;
          archive.load_from(std::string(filename));
          for (int i = 0; i < ntensors; ++i) {
            torch::Tensor *tensor_ptr = tensor_ptr_from_ocaml(tensors[i]);
            if (tensor_ptr->device().type() == at::kCPU)
              archive.read(std::string(tensor_names[i]), *tensor_ptr);
            else {
              torch::Tensor tmp_tensor =
                  torch::empty_like(*tensor_ptr, at::device(at::kCPU));
              archive.read(std::string(tensor_names[i]), tmp_tensor);
              tensor_ptr->copy_(tmp_tensor);
            }
          })
}

raw_tensor at_load(char *filename) {
  PROTECT(torch::Tensor tensor; torch::load(tensor, filename);
          return tensor_to_ocaml(tensor);)
  return nullptr;
}

int at_get_num_interop_threads() {
  PROTECT(return at::get_num_interop_threads();)
  return -1;
}

int at_get_num_threads() {
  PROTECT(return at::get_num_threads();)
  return -1;
}

void at_set_num_interop_threads(int n_threads) {
  PROTECT(at::set_num_interop_threads(n_threads);)
}

void at_set_num_threads(int n_threads) {
  PROTECT(at::set_num_threads(n_threads);)
}

void at_free(gc_tensor t) { delete (tensor_ptr_from_ocaml(t)); }

void at_run_backward(gc_tensor *tensors, int ntensors, gc_tensor *inputs,
                     int ninputs, raw_tensor *outputs, int keep_graph,
                     int create_graph) {
  PROTECT(
      vector<torch::autograd::Edge> roots;
      for (int i = 0; i < ntensors; ++i) roots.push_back(
          torch::autograd::impl::gradient_edge(tensor_from_ocaml(tensors[i])));

      vector<torch::autograd::Edge> inputs_; for (int i = 0; i < ninputs; ++i) {
        torch::Tensor *input_ = tensor_ptr_from_ocaml(inputs[i]);
        if (!input_->requires_grad())
          throw std::invalid_argument(
              "one of the input tensor does not use set_requires_grad");
        inputs_.push_back(torch::autograd::impl::gradient_edge(*input_));
      }

      vector<torch::autograd::Variable>
          grads;
      for (int i = 0; i < ntensors; ++i)
          grads.push_back(torch::ones_like(tensor_from_ocaml(tensors[i])));

      caml_release_runtime_system(); torch::autograd::variable_list vl; try {
        vl = torch::autograd::Engine::get_default_engine().execute(
            roots, grads, keep_graph, create_graph, false, inputs_);
      } catch (const exception &) {
        caml_acquire_runtime_system();
        throw;
      } caml_acquire_runtime_system();
      for (int i = 0; i < ninputs;
           ++i) { outputs[i] = tensor_to_ocaml(vl[i]); })
}

optimizer ato_adam(double learning_rate, double beta1, double beta2,
                   double weight_decay, double eps) {
  PROTECT(auto options = torch::optim::AdamOptions(learning_rate)
                             .betas(std::tuple<double, double>(beta1, beta2))
                             .weight_decay(weight_decay)
                             .eps(eps);
          return new torch::optim::Adam(vector<torch::Tensor>(), options);)
  return nullptr;
}

optimizer ato_rmsprop(double learning_rate, double alpha, double eps,
                      double weight_decay, double momentum, int centered) {
  PROTECT(auto options = torch::optim::RMSpropOptions(learning_rate)
                             .alpha(alpha)
                             .eps(eps)
                             .weight_decay(weight_decay)
                             .momentum(momentum)
                             .centered(centered != 0);
          return new torch::optim::RMSprop(vector<torch::Tensor>(), options);)
  return nullptr;
}

optimizer ato_sgd(double learning_rate, double momentum, double dampening,
                  double weight_decay, int nesterov) {
  PROTECT(auto options = torch::optim::SGDOptions(learning_rate)
                             .momentum(momentum)
                             .dampening(dampening)
                             .weight_decay(weight_decay)
                             .nesterov(nesterov);
          return new torch::optim::SGD(vector<torch::Tensor>(), options);)
  return nullptr;
}

void ato_add_parameters(optimizer t, gc_tensor *tensors, int ntensors) {
  PROTECT(for (int i = 0; i < ntensors; ++i) t->param_groups()[0]
              .params()
              .push_back(tensor_from_ocaml(tensors[i]));)
}

template <class T> void set_lr(optimizer t, double learning_rate) {
  torch::optim::OptimizerOptions *d = &(t->defaults());
  if (auto p = dynamic_cast<T *>(d)) {
    p->lr(learning_rate);
    for (auto &param_group : t->param_groups()) {
      torch::optim::OptimizerOptions *d = &(param_group.options());
      if (auto p2 = dynamic_cast<T *>(d)) {
        p2->lr(learning_rate);
      } else
        throw std::invalid_argument("unexpected param group type");
    }
  }
}

void ato_set_learning_rate(optimizer t, double learning_rate) {
  PROTECT(set_lr<torch::optim::AdamOptions>(t, learning_rate);
          set_lr<torch::optim::AdamWOptions>(t, learning_rate);
          set_lr<torch::optim::RMSpropOptions>(t, learning_rate);
          set_lr<torch::optim::SGDOptions>(t, learning_rate);)
}

template <class T>
void set_lr_group(optimizer t, size_t group, double learning_rate) {
  auto &param_group = t->param_groups().at(group);
  torch::optim::OptimizerOptions *d = &(param_group.options());
  if (auto p = dynamic_cast<T *>(d)) {
    p->lr(learning_rate);
  }
}

void ato_set_learning_rate_group(optimizer t, size_t group,
                                 double learning_rate) {
  PROTECT(set_lr_group<torch::optim::AdamOptions>(t, group, learning_rate);
          set_lr_group<torch::optim::AdamWOptions>(t, group, learning_rate);
          set_lr_group<torch::optim::RMSpropOptions>(t, group, learning_rate);
          set_lr_group<torch::optim::SGDOptions>(t, group, learning_rate);)
}

void ato_set_momentum(optimizer t, double momentum) {
  PROTECT(
      torch::optim::OptimizerOptions *d = &(t->defaults());
      if (auto adam = dynamic_cast<torch::optim::AdamOptions *>(d)) {
        auto betas = adam->betas();
        adam->betas(std::tuple<double, double>(momentum, get<1>(betas)));
        for (auto &param_group : t->param_groups()) {
          torch::optim::OptimizerOptions *d = &(param_group.options());
          if (auto adam2 = dynamic_cast<torch::optim::AdamOptions *>(d)) {
            adam2->betas(std::tuple<double, double>(momentum, get<1>(betas)));
          } else
            throw std::invalid_argument("unexpected param group type");
        }
      } else if (auto adamw = dynamic_cast<torch::optim::AdamWOptions *>(d)) {
        auto betas = adamw->betas();
        adamw->betas(std::tuple<double, double>(momentum, get<1>(betas)));
        for (auto &param_group : t->param_groups()) {
          torch::optim::OptimizerOptions *d = &(param_group.options());
          if (auto adamw2 = dynamic_cast<torch::optim::AdamWOptions *>(d)) {
            adamw2->betas(std::tuple<double, double>(momentum, get<1>(betas)));
          } else
            throw std::invalid_argument("unexpected param group type");
        }
      } else if (auto rms = dynamic_cast<torch::optim::RMSpropOptions *>(d)) {
        rms->momentum(momentum);
        for (auto &param_group : t->param_groups()) {
          torch::optim::OptimizerOptions *d = &(param_group.options());
          if (auto rms2 = dynamic_cast<torch::optim::RMSpropOptions *>(d)) {
            rms2->momentum(momentum);
          } else
            throw std::invalid_argument("unexpected param group type");
        }
      } else if (auto sgd = dynamic_cast<torch::optim::SGDOptions *>(d)) {
        sgd->momentum(momentum);
        for (auto &param_group : t->param_groups()) {
          torch::optim::OptimizerOptions *d = &(param_group.options());
          if (auto sgd2 = dynamic_cast<torch::optim::SGDOptions *>(d)) {
            sgd2->momentum(momentum);
          } else
            throw std::invalid_argument("unexpected param group type");
        }
      } else throw std::invalid_argument("unexpected optimizer");)
}

void ato_set_momentum_group(optimizer t, size_t group, double momentum) {
  PROTECT(
      auto &param_group = t->param_groups().at(group);
      torch::optim::OptimizerOptions *d = &(param_group.options());

      if (auto adam = dynamic_cast<torch::optim::AdamOptions *>(d)) {
        auto betas = adam->betas();
        adam->betas(std::tuple<double, double>(momentum, get<1>(betas)));
      } else if (auto adamw = dynamic_cast<torch::optim::AdamWOptions *>(d)) {
        auto betas = adamw->betas();
        adamw->betas(std::tuple<double, double>(momentum, get<1>(betas)));
      } else if (auto rms = dynamic_cast<torch::optim::RMSpropOptions *>(d)) {
        rms->momentum(momentum);
      } if (auto sgd = dynamic_cast<torch::optim::SGDOptions *>(d)) {
        sgd->momentum(momentum);
      } else throw std::invalid_argument("unexpected optimizer");)
}

template <class T> void set_weight_decay(optimizer t, double weight_decay) {
  torch::optim::OptimizerOptions *d = &(t->defaults());
  if (auto p = dynamic_cast<T *>(d)) {
    p->weight_decay(weight_decay);
    for (auto &param_group : t->param_groups()) {
      torch::optim::OptimizerOptions *d = &(param_group.options());
      if (auto p2 = dynamic_cast<T *>(d)) {
        p2->weight_decay(weight_decay);
      } else
        throw std::invalid_argument("unexpected param group type");
    }
  }
}

void ato_set_weight_decay(optimizer t, double weight_decay) {
  PROTECT(set_weight_decay<torch::optim::AdamOptions>(t, weight_decay);
          set_weight_decay<torch::optim::AdamWOptions>(t, weight_decay);
          set_weight_decay<torch::optim::RMSpropOptions>(t, weight_decay);
          set_weight_decay<torch::optim::SGDOptions>(t, weight_decay);)
}

template <class T>
void set_weight_decay_group(optimizer t, size_t group, double weight_decay) {
  auto &param_group = t->param_groups().at(group);
  torch::optim::OptimizerOptions *d = &(param_group.options());
  if (auto p = dynamic_cast<T *>(d)) {
    p->weight_decay(weight_decay);
  }
}

void ato_set_weight_decay_group(optimizer t, size_t group,
                                double weight_decay) {
  PROTECT(
      set_weight_decay_group<torch::optim::AdamOptions>(t, group, weight_decay);
      set_weight_decay_group<torch::optim::AdamWOptions>(t, group,
                                                         weight_decay);
      set_weight_decay_group<torch::optim::RMSpropOptions>(t, group,
                                                           weight_decay);
      set_weight_decay_group<torch::optim::SGDOptions>(t, group, weight_decay);)
}

void ato_zero_grad(optimizer t) { PROTECT(t->zero_grad();) }

void ato_step(optimizer t) { PROTECT(t->step();) }

void ato_free(optimizer t) { delete (t); }

scalar ats_int(int64_t v) {
  PROTECT(return new torch::Scalar(v);)
  return nullptr;
}

scalar ats_float(double v) {
  PROTECT(return new torch::Scalar(v);)
  return nullptr;
}

int64_t ats_to_int(scalar s) {
  PROTECT(return s->toLong();)
  return -1;
}

double ats_to_float(scalar s) {
  PROTECT(return s->toDouble();)
  return 0.;
}

char *ats_to_string(scalar s) {
  PROTECT(using namespace at; std::ostringstream oss; oss << (*s);
          return strdup(oss.str().c_str());)
  return nullptr;
}

void ats_free(scalar s) { delete (s); }

int atc_cuda_device_count() {
  PROTECT(return torch::cuda::device_count();)
  return -1;
}

int atc_cuda_is_available() {
  PROTECT(return torch::cuda::is_available();)
  return -1;
}

int atc_cudnn_is_available() {
  PROTECT(return torch::cuda::cudnn_is_available();)
  return -1;
}

void atc_set_benchmark_cudnn(int b) {
  at::globalContext().setBenchmarkCuDNN(b);
}

module atm_load(char *filename, int device) {
  PROTECT(return new torch::jit::script::Module(torch::jit::load(
      filename, optional_device_of_int(device)));)
  return nullptr;
}

module atm_load_str(char *data, size_t sz, int device) {
  PROTECT(std::istringstream stream(std::string(data, sz));
          return new torch::jit::script::Module(
              torch::jit::load(stream, optional_device_of_int(device)));)
  return nullptr;
}

raw_tensor atm_forward(module m, gc_tensor *tensors, int ntensors) {
  PROTECT(
      std::vector<torch::jit::IValue> inputs;
      for (int i = 0; i < ntensors; ++i)
          inputs.push_back(tensor_from_ocaml(tensors[i]));
      caml_release_runtime_system(); torch::jit::IValue output;

      // In case of exception, we need to re-acquire the runtime
      // lock before re-raising, since PROTECT re-enters ocaml.
      try {
        output = m->forward(std::move(inputs));
      } catch (const exception &) {
        caml_acquire_runtime_system();
        throw;
      }

      caml_acquire_runtime_system();
      if (!output.isTensor()) throw std::invalid_argument(
          "forward did not return a tensor");
      return tensor_to_ocaml(output.toTensor());)
  return nullptr;
}

ivalue atm_forward_(module m, ivalue *ivalues, int nivalues) {
  PROTECT(
      std::vector<torch::jit::IValue> inputs;
      for (int i = 0; i < nivalues; ++i) inputs.push_back(*(ivalues[i]));
      caml_release_runtime_system(); torch::jit::IValue output;

      // In case of exception, we need to re-acquire the runtime
      // lock before re-raising, since PROTECT re-enters ocaml.
      try { output = m->forward(inputs); } catch (const exception &) {
        caml_acquire_runtime_system();
        throw;
      }

      caml_acquire_runtime_system();
      return new torch::jit::IValue(output);)
  return nullptr;
}
// To return this OrderedDict<string, Tensor>, we pass it a tuple
// IValue containing
// * list of strings IValue (names)
// * list of tensors IValue (tensors)
ivalue atm_named_buffers(module m) {
  PROTECT(c10::List<torch::jit::IValue> names(c10::StringType::get());
          c10::List<torch::jit::IValue> tensors(c10::TensorType::get());
          vector<torch::jit::IValue> names_and_tensors;
          for (auto b
               : m->named_buffers()) {
            names.push_back(torch::jit::IValue(b.name));
            tensors.push_back(torch::jit::IValue(b.value));
          };
          names_and_tensors.push_back(
              torch::jit::IValue(c10::List<torch::jit::IValue>(names)));
          names_and_tensors.push_back(
              torch::jit::IValue(c10::List<torch::jit::IValue>(tensors)));
          return new torch::jit::IValue(
              torch::ivalue::Tuple::create(names_and_tensors));)
  return nullptr;
}

void atm_free(module m) { delete (m); }

void atm_to(module m, int device, int dtype, bool non_blocking) {
  PROTECT(m->to(device_of_int(device), at::ScalarType(dtype), non_blocking);)
}

ivalue ati_tensor(gc_tensor t) {
  PROTECT(return new torch::jit::IValue(tensor_from_ocaml(t));)
  return nullptr;
}

ivalue ati_int(int64_t i) {
  PROTECT(return new torch::jit::IValue(i);)
  return nullptr;
}

ivalue ati_double(double d) {
  PROTECT(return new torch::jit::IValue(d);)
  return nullptr;
}

ivalue ati_bool(int i) {
  PROTECT(return new torch::jit::IValue((bool)i);)
  return nullptr;
}

ivalue ati_string(char *s) {
  PROTECT(string str(s); return new torch::jit::IValue(str);)
  return nullptr;
}

ivalue ati_none() {
  PROTECT(return new torch::jit::IValue();)
  return nullptr;
}

ivalue ati_tuple(ivalue *is, int nvalues) {
  PROTECT(vector<torch::jit::IValue> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
          return new torch::jit::IValue(torch::ivalue::Tuple::create(vec));)
  return nullptr;
}

ivalue ati_generic_list(ivalue *is, int nvalues) {
  PROTECT(c10::List<torch::jit::IValue> vec(c10::AnyType::get());
          for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
          return new torch::jit::IValue(c10::List<torch::jit::IValue>(vec));)
  return nullptr;
}

ivalue ati_generic_dict(ivalue *is, int nvalues) {
  c10::Dict<torch::jit::IValue, torch::jit::IValue> dict(c10::AnyType::get(),
                                                         c10::AnyType::get());
  PROTECT(for (int i = 0; i < nvalues; ++i)
              dict.insert(*(is[2 * i]), *(is[2 * i + 1]));
          return new torch::jit::IValue(dict);)
  return nullptr;
}

ivalue ati_int_list(int64_t *is, int nvalues) {
  PROTECT(c10::List<int64_t> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(is[i]);
          return new torch::jit::IValue(vec);)
  return nullptr;
}

ivalue ati_double_list(double *is, int nvalues) {
  PROTECT(c10::List<double> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(is[i]);
          return new torch::jit::IValue(vec);)
  return nullptr;
}

ivalue ati_bool_list(char *is, int nvalues) {
  PROTECT(c10::List<bool> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(is[i] != 0);
          return new torch::jit::IValue(vec);)
  return nullptr;
}

ivalue ati_string_list(char **is, int nvalues) {
  PROTECT(c10::List<string> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(string(is[i]));
          return new torch::jit::IValue(vec);)
  return nullptr;
}

ivalue ati_tensor_list(gc_tensor *is, int nvalues) {
  PROTECT(c10::List<at::Tensor> vec;
          for (int i = 0; i < nvalues; ++i)
              vec.push_back(tensor_from_ocaml(is[i]));
          return new torch::jit::IValue(vec);)
  return nullptr;
}

int ati_tag(ivalue i) {
  PROTECT(
      if (i->isNone()) return 0; else if (i->isTensor()) return 1;
      else if (i->isDouble()) return 2; else if (i->isInt()) return 3;
      else if (i->isBool()) return 4; else if (i->isTuple()) return 5;
      else if (i->isIntList()) return 6; else if (i->isDoubleList()) return 7;
      else if (i->isBoolList()) return 8; else if (i->isString()) return 9;
      else if (i->isTensorList()) return 10; else if (i->isList()) return 12;
      else if (i->isGenericDict()) return 13; else if (i->isObject()) return 14;
      throw std::invalid_argument(("unsupported tag " + i->tagKind()).c_str());
      return -1;)
  return -1;
}

int64_t ati_to_int(ivalue i) {
  PROTECT(return i->toInt();)
  return -1;
}

double ati_to_double(ivalue i) {
  PROTECT(return i->toDouble();)
  return 0;
}

int ati_to_bool(ivalue i) {
  PROTECT(return i->toBool();)
  return -1;
}

char *ati_to_string(ivalue i) {
  PROTECT(auto str = i->toStringRef(); return strdup(str.c_str());)
  return nullptr;
}

raw_tensor ati_to_tensor(ivalue i) {
  PROTECT(return tensor_to_ocaml(i->toTensor());)
  return nullptr;
}

int ati_length(ivalue i) {
  PROTECT(if (i->isTuple()) return i->toTuple()->elements().size();
          else if (i->isIntList()) return i->toIntList().size();
          else if (i->isDoubleList()) return i->toDoubleList().size();
          else if (i->isBoolList()) return i->toBoolList().size();
          else if (i->isString()) return i->toStringRef().size();
          else if (i->isTensorList()) return i->toTensorList().size();
          else if (i->isList()) return i->toList().size();
          else if (i->isGenericDict()) return i->toGenericDict().size();
          throw std::invalid_argument(
              ("unsupported tag for length " + i->tagKind()).c_str());
          return -1;)
  return -1;
}

int ati_tuple_length(ivalue i) {
  PROTECT(return i->toTuple()->elements().size();)
  return -1;
}

void ati_to_tuple(ivalue i, ivalue *outputs, int noutputs) {
  PROTECT(auto vec = i->toTuple()->elements(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected tuple size");
  } for (int i = 0; i < noutputs;
         ++i) outputs[i] = new torch::jit::IValue(vec[i]);)
}

int ati_list_length(ivalue i) {
  PROTECT(return i->toList().size();)
  return -1;
}

void ati_to_generic_list(ivalue i, ivalue *outputs, int noutputs) {
  PROTECT(auto vec = i->toList(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected list size");
  } for (int i = 0; i < noutputs; ++i) outputs[i] =
                                      new torch::jit::IValue(vec[i]);)
}

void ati_to_generic_dict(ivalue i, ivalue *outputs, int noutputs) {
  PROTECT(
      auto dict = i->toGenericDict(); if (dict.size() != noutputs) {
        throw std::invalid_argument("unexpected dict size");
      } int k = 0;
      for (auto it = dict.begin(); it != dict.end(); ++it) {
        outputs[k++] = new torch::jit::IValue(it->key());
        outputs[k++] = new torch::jit::IValue(it->value());
      })
}

void ati_to_int_list(ivalue i, int64_t *outputs, int noutputs) {
  PROTECT(auto vec = i->toIntList(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected list<int> size");
  } for (int i = 0; i < noutputs; ++i) outputs[i] = vec[i];)
}

void ati_to_double_list(ivalue i, double *outputs, int noutputs) {
  PROTECT(auto vec = i->toDoubleList(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected list<double> size");
  } for (int i = 0; i < noutputs; ++i) outputs[i] = vec[i];)
}

void ati_to_bool_list(ivalue i, char *outputs, int noutputs) {
  PROTECT(auto vec = i->toBoolList(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected list<bool> size");
  } for (int i = 0; i < noutputs; ++i) outputs[i] = vec[i];)
}

void ati_to_tensor_list(ivalue i, raw_tensor *outputs, int noutputs) {
  PROTECT(auto vec = i->toTensorList(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected list<tensor> size");
  } for (int i = 0; i < noutputs; ++i) outputs[i] = tensor_to_ocaml(vec[i]);)
}

ivalue ati_object_method_(ivalue i, char *method_name, ivalue *ivalues,
                          int nivalues) {
  PROTECT(std::vector<torch::jit::IValue> inputs;
          inputs.push_back(*i); // self parameter
          for (int j = 0; j < nivalues; ++j) inputs.push_back(*(ivalues[j]));
          torch::jit::IValue output = i->toObjectRef().type()->getMethod(
              method_name)(std::move(inputs));
          return new torch::jit::IValue(output);)
  return nullptr;
}

ivalue ati_object_getattr_(ivalue i, char *attr_name) {
  PROTECT(torch::jit::IValue output = i->toObjectRef().getAttr(attr_name);
          return new torch::jit::IValue(output);)
  return nullptr;
}

ivalue ati_clone(ivalue i) {
  PROTECT(return new torch::jit::IValue(*i);)
  return nullptr;
}

void ati_free(ivalue i) { delete (i); }

void at_set_graph_executor_optimize(bool o) {
  torch::jit::setGraphExecutorOptimize(o);
}

#include "torch_api_generated.cpp"

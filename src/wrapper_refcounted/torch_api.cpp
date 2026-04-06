#include <torch/csrc/autograd/engine.h>
#include <torch/torch.h>
#include <ATen/autocast_mode.h>
#include <torch/script.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#include <torch/csrc/cuda/memory_snapshot.h>
#include <stdexcept>
#include <vector>
#include <caml/fail.h>
#include <caml/memory.h>
#undef invalid_argument
#include <cuda_runtime.h>
#include "torch_api.h"
#include <caml/mlvalues.h>
#include <caml/alloc.h>
#include <caml/threads.h>
#include <caml/bigarray.h>
#include "ocaml_runtime_props.h"
#include "ocaml_to_cpp.h"

// Following the conventions suggested in the link below, I named all functions that don't
// allocate as noalloc and don't use CAMLparam and CAMLreturn in them.
// https://blog.janestreet.com/faster-ocaml-to-c-calls/
//
// In theory I don't care about C++ allocations and could also not use CAMLparam in
// functions that use those, but that feels too clever. Also if things are somewhat
// complicated it's best to err on the side of doing things properly because you have to
// reason about all functions that get called, even indirectly.

using namespace std;

raw_tensor get_raw_tensor_noalloc(value managed_tensor) {
  return pointer_of_custom_val<torch::TensorImpl>(managed_tensor);
}

void increment_refcount_internal_noalloc(value managed_tensor) {
  raw_tensor t = get_raw_tensor_noalloc(managed_tensor);

  // Increase refcount by creating a wrapper, then release it (keeping refcount the same)
  // and ignore the resulting pointer
  auto ptr =
      c10::intrusive_ptr<torch::TensorImpl, torch::UndefinedTensorImpl>::reclaim_copy(t);

  ptr.release();
}

void decrement_refcount_internal(value managed_tensor) {
  CAMLparam1(managed_tensor);
  // There are three types here that can hold a pointer to TensorImpl: managed_tensor
  // (with its raw_tensor field), c10::intrusive_ptr, and torch::Tensor. This function
  // passes the same pointer from managed_tensor to c10::intrusive_ptr then to
  // torch::Tensor. torch::Tensor decrements the refcount on TensorImpl when its
  // destructor is called.

  // Get an intrusive ptr without incrementing TensorImpl refcount
  raw_tensor t = get_raw_tensor_noalloc(managed_tensor);
  auto ptr =
      c10::intrusive_ptr<torch::TensorImpl, torch::UndefinedTensorImpl>::reclaim(t);
  bool last_reference = ptr.use_count() == 1;
#if OCAML_DEPENDENT_MEM_TRACKING
  // This does a std::move on [ptr] and takes ownership of the pointer to TensorImpl. This
  // means [ptr] will point to null after this line.
  torch::Tensor tensor = torch::Tensor(std::move(ptr));

  // tell OCaml GC that the off-heap dependent memory is going away
  unsigned long int off_heap_cpu_memory_bytes = 0;
  if (tensor.defined() && tensor.device() == at::kCPU && last_reference &&
      tensor.storage().use_count() == 1) {
    off_heap_cpu_memory_bytes = tensor.numel() * tensor.element_size();
    caml_free_dependent_memory(managed_tensor, off_heap_cpu_memory_bytes);
  }
#endif
  // If we were holding the last reference, change the pointer in the OCaml allocation to
  // a nullptr. This turns use-after-free into deterministic segfaults.
  if (last_reference) {
    *static_cast<void **>(Data_custom_val(managed_tensor)) = nullptr;
  }
  CAMLreturn0;
}

int get_refcount_internal_noalloc(value managed_tensor) {
  raw_tensor t = get_raw_tensor_noalloc(managed_tensor);
  // This will not increment
  auto ptr =
      c10::intrusive_ptr<torch::TensorImpl, torch::UndefinedTensorImpl>::reclaim(t);
  int refcount = ptr.use_count();
  // This will not decrement
  ptr.release();
  return refcount;
}

torch::Tensor tensor_from_ocaml(gc_tensor t) {
  // invoke reclaim_copy to increment the refcount by 1 (until this new Tensor
  // gets dropped by C++)
  auto ptr =
      c10::intrusive_ptr<torch::TensorImpl, torch::UndefinedTensorImpl>::reclaim_copy(t);
  return torch::Tensor(ptr);
}

// Same as tensor_from_ocaml, but does not increment refcount (for use in finaliser)
torch::Tensor take_tensor_from_ocaml(gc_tensor t) {
  auto ptr =
      c10::intrusive_ptr<torch::TensorImpl, torch::UndefinedTensorImpl>::reclaim(t);
  return torch::Tensor(ptr);
}

// It's actually better to pass tensors by reference, sparing some atomic
// refcount updates. By doing so, newly created tensors passed to this function
// should have a refcount of exactly 1.
// https://dev-discuss.pytorch.org/t/we-shouldnt-feel-bad-about-passing-tensor-by-reference/85
raw_tensor tensor_to_ocaml(const torch::Tensor &cpp_tensor) {
  auto ptr = cpp_tensor.getIntrusivePtr();
  return ptr.release();
}

torch::Scalar *scalar_from_ocaml_noalloc(value scalar) {
  return pointer_of_custom_val<torch::Scalar>(scalar);
}

std::optional<torch::Scalar> scalar_option_from_ocaml(value scalar_opt) {
  CAMLparam1(scalar_opt);
  CAMLlocal1(caml_scalar);
  std::optional<torch::Scalar> result = std::nullopt;

  if (Is_some(scalar_opt)) {
    caml_scalar = Some_val(scalar_opt);
    result = *scalar_from_ocaml_noalloc(caml_scalar);
  }

  CAMLreturnT(std::optional<torch::Scalar>, result);
}

torch::Tensor rc_tensor_from_ocaml(value tensor) {
  CAMLparam1(tensor);
  auto ptr = pointer_of_custom_val<torch::TensorImpl>(tensor);
  auto torch_tensor = torch::Tensor(
      c10::intrusive_ptr<torch::TensorImpl, torch::UndefinedTensorImpl>::reclaim_copy(
          ptr));
  CAMLreturnT(torch::Tensor, torch_tensor);
}

value rc_tensor_to_ocaml(const torch::Tensor &tensor) {
  CAMLparam0();
  CAMLlocal1(tensor_caml);
  tensor_caml = pointer_to_custom_val(tensor_to_ocaml(tensor));
#if OCAML_DEPENDENT_MEM_TRACKING
  if (tensor.defined() && tensor.device() == at::kCPU &&
      tensor.getIntrusivePtr().use_count() /* getIntrusivePtr call here adds one to the
                                              use_count so we check for 2 */
          == 2 &&
      tensor.storage().use_count() == 1) {
    unsigned long int off_heap_cpu_memory_bytes = tensor.numel() * tensor.element_size();
    if (off_heap_cpu_memory_bytes) {
      caml_alloc_dependent_memory(tensor_caml, off_heap_cpu_memory_bytes);
    }
  }
#endif

  CAMLreturn(tensor_caml);
}

value prepare_ocaml_tensor(const torch::Tensor &tensor) {
  return rc_tensor_to_ocaml(tensor);
}

std::optional<torch::Tensor> tensor_option_from_ocaml(value tensor_option) {
  CAMLparam1(tensor_option);
  CAMLlocal1(tensor_caml);
  std::optional<torch::Tensor> result;

  if (Is_some(tensor_option)) {
    tensor_caml = Some_val(tensor_option);
    result = rc_tensor_from_ocaml(tensor_caml);
  }

  CAMLreturnT(std::optional<torch::Tensor>, result);
}

std::vector<torch::Tensor> of_ocaml_tensor_list(value tensor_list) {
  CAMLparam1(tensor_list);
  CAMLlocal1(list_ptr);
  list_ptr = tensor_list;
  std::vector<torch::Tensor> result;
  result.reserve(list_length_noalloc(list_ptr));

  while (list_ptr != Val_emptylist) {
    result.push_back(rc_tensor_from_ocaml(Field(list_ptr, 0)));
    list_ptr = Field(list_ptr, 1);
  }

  CAMLreturnT(std::vector<torch::Tensor>, result);
}

value to_ocaml_tensor_list(const std::vector<torch::Tensor> &tensors) {
  CAMLparam0();
  CAMLlocal3(curr_cell, prev_cell, curr_tensor);
  prev_cell = Val_emptylist;

  for (auto it = tensors.rbegin(); it != tensors.rend(); ++it) {
    curr_tensor = rc_tensor_to_ocaml(*it);
    curr_cell = caml_alloc_small(2, 0);
    Field(curr_cell, 0) = curr_tensor;
    Field(curr_cell, 1) = prev_cell;
    prev_cell = curr_cell;
  }

  CAMLreturn(prev_cell);
}

void at_manual_seed(int64_t seed) { torch::manual_seed(seed); }

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

value at_new_tensor() {
  CAMLparam0();
  PROTECT(CAMLreturn(rc_tensor_to_ocaml(torch::Tensor()));)
}

value at_tensor_of_data(value vs, value dims, int element_size_in_bytes, int type) {
  CAMLparam2(vs, dims);
  PROTECT(void *vs_data = Caml_ba_data_val(vs);
          std::vector<int64_t> dims_vec = vec_of_ocaml_int_list(dims);
          torch::Tensor tensor = torch::zeros(dims_vec, torch::ScalarType(type));
          if ((int64_t)element_size_in_bytes != tensor.element_size()) throw std::
              invalid_argument("incoherent element sizes in bytes");
          void *tensor_data = tensor.data_ptr();
          memcpy(tensor_data, vs_data, tensor.numel() * element_size_in_bytes);
          CAMLreturn(rc_tensor_to_ocaml(tensor));)
}

template <typename... Args> std::string sstr(Args &&...args) {
  std::ostringstream sstr;
  // fold expression
  ((sstr << std::dec) << ... << args);
  return sstr.str();
}

static void copy_to(torch::Tensor tensor, void *vs, size_t copy_size) {
  if (tensor.device().type() != at::kCPU) {
    // First, move the tensor to the CPU
    torch::Tensor tmp_tensor = tensor.to(at::kCPU).contiguous();
    void *tensor_data = tmp_tensor.data_ptr();
    memcpy(vs, tensor_data, copy_size);
  } else {
    // Make sure the tensor is contiguous before copying
    auto tmp_tensor = tensor.contiguous();
    void *tensor_data = tmp_tensor.data_ptr();
    memcpy(vs, tensor_data, copy_size);
  }
}

void at_copy_to_elements(value t, value vs, int64_t numel, int elt_size_in_bytes) {
  CAMLparam2(t, vs);
  PROTECT(torch::Tensor tensor = rc_tensor_from_ocaml(t);
          void *vs_data = Caml_ba_data_val(vs);
          if (elt_size_in_bytes != 0 &&
              (int64_t)elt_size_in_bytes != tensor.element_size()) throw std::
              invalid_argument(sstr("incoherent element sizes in bytes: dst (",
                                    elt_size_in_bytes, ") != src (",
                                    tensor.element_size(), ")"));
          if ((int64_t)numel > tensor.numel()) throw std::invalid_argument(
              sstr("target numel (", numel, ") is larger than tensor numel (",
                   tensor.numel(), ")"));
          copy_to(tensor, vs_data, elt_size_in_bytes * numel);

          CAMLreturn0;)
}

void at_copy_to_bytes(value t, value bytes, int64_t bytes_offset, int64_t bytes_len) {
  CAMLparam2(t, bytes);
  PROTECT(torch::Tensor tensor = rc_tensor_from_ocaml(t);
          char *bytes_data = (char *)Caml_ba_data_val(bytes);
          size_t tensor_total_size = tensor.numel() * tensor.element_size();
          if (bytes_len != tensor_total_size) throw std::invalid_argument(
              sstr("bytes is not the correct length for this tensor: ", tensor_total_size,
                   " != ", bytes_len));
          copy_to(tensor, bytes_data + bytes_offset, tensor_total_size);

          CAMLreturn0;)
}

void at_copy_from_bytes(value t, value bytes, int64_t bytes_offset, int64_t bytes_len) {
  CAMLparam2(t, bytes);
  PROTECT(torch::Tensor tensor = rc_tensor_from_ocaml(t);

          auto dtype = tensor.dtype();

          char *bytes_data = (char *)Caml_ba_data_val(bytes);
          int64_t need_bytes = tensor.numel() * tensor.element_size();
          if (need_bytes != bytes_len) throw std::invalid_argument(
              sstr("bytes is not the correct length for this tensor: ", need_bytes,
                   " != ", bytes_len));
          torch::Tensor tmp_tensor =
              torch::from_blob(bytes_data + bytes_offset, tensor.sizes(),
                               torch::TensorOptions().dtype(dtype));
          tensor.copy_(tmp_tensor, 0);

          CAMLreturn0;)
}

value at_float_vec(value values, int type) {
  CAMLparam1(values);
  PROTECT(std::vector<double> vs = vec_of_ocaml_double_list(values);
          size_t len = vs.size();

          torch::Tensor tensor =
              torch::empty({static_cast<int>(len)}, torch::ScalarType(type));
          for (int i = 0; i < len; ++i) { tensor[i] = vs[i]; }

          CAMLreturn(rc_tensor_to_ocaml(tensor));)
}

value at_int_vec(value values, int type) {
  CAMLparam1(values);
  PROTECT(std::vector<int64_t> vs = vec_of_ocaml_int_list(values);

          size_t len = vs.size();

          torch::Tensor tensor =
              torch::empty({static_cast<int>(len)}, torch::ScalarType(type));
          for (int i = 0; i < len; ++i) tensor[i] = vs[i];
          CAMLreturn(rc_tensor_to_ocaml(tensor));)
}

int at_defined(value t) {
  CAMLparam1(t);
  PROTECT(CAMLreturnT(int, rc_tensor_from_ocaml(t).defined());)
}

int at_is_sparse(value t) {
  CAMLparam1(t);
  PROTECT(CAMLreturnT(int, rc_tensor_from_ocaml(t).is_sparse());)
}

int at_dim(value t) {
  CAMLparam1(t);
  PROTECT(CAMLreturnT(int, rc_tensor_from_ocaml(t).dim());)
}

value at_shape(value t) {
  CAMLparam1(t);
  PROTECT(torch::Tensor tensor = rc_tensor_from_ocaml(t);
          CAMLreturn(ocaml_list_of_ints(tensor.sizes()));)
}

int at_scalar_type(value t) {
  CAMLparam1(t);
  PROTECT(CAMLreturnT(int, static_cast<int>(rc_tensor_from_ocaml(t).scalar_type()));)
}
int at_use_count(value t) {
  CAMLparam1(t);
  PROTECT(CAMLreturnT(int, static_cast<int>(rc_tensor_from_ocaml(t).use_count()));)
}

void at_autocast_clear_cache() { at::autocast::clear_cache(); }

int at_autocast_decrement_nesting() { PROTECT(return at::autocast::decrement_nesting();) }

int at_autocast_increment_nesting() { PROTECT(return at::autocast::increment_nesting();) }

int at_autocast_is_enabled() {
  PROTECT(return at::autocast::is_autocast_enabled(at::kCUDA);)
}

int at_autocast_set_enabled(int b) {
  PROTECT(bool is_enabled = at::autocast::is_autocast_enabled(at::kCUDA);
          at::autocast::set_autocast_enabled(at::kCUDA, b);

          return is_enabled;)
}

int at_device(value t) {
  CAMLparam1(t);
  PROTECT(auto device = rc_tensor_from_ocaml(t).device();

          if (device.is_cpu()) return -1;

          CAMLreturnT(int, device.index());)
}

void at_backward(value t, int keep_graph, int create_graph) {
  CAMLparam1(t);
  PROTECT(
      // Need to be careful about the order. We have to call
      // [tensor_from_ocaml] before releasing the runtime lock because
      // at this point nothing inside of ocaml is keeping this alive
      // any more.

      auto tensor = rc_tensor_from_ocaml(t);

      caml_release_runtime_system();
      try { tensor.backward({}, keep_graph, create_graph); } catch (const exception &) {
        caml_acquire_runtime_system();
        throw;
      } caml_acquire_runtime_system();
      CAMLreturn0;)
}

int at_requires_grad(value t) {
  CAMLparam1(t);
  PROTECT(CAMLreturnT(int, rc_tensor_from_ocaml(t).requires_grad());)
}

int at_grad_set_enabled(int b) {
  PROTECT(bool is_enabled = torch::autograd::GradMode::is_enabled();
          torch::autograd::GradMode::set_enabled(b);

          return is_enabled;)
}

value at_get(value t, int index) {
  CAMLparam1(t);
  PROTECT(CAMLreturn(rc_tensor_to_ocaml(rc_tensor_from_ocaml(t)[index]));)
}

template <typename T> T at_value_at_indexes(value t, value indexes) {
  CAMLparam2(t, indexes);
  PROTECT(torch::Tensor tensor = rc_tensor_from_ocaml(t);

          while (indexes != Val_emptylist) {
            tensor = tensor[Int_val(Field(indexes, 0))];
            indexes = Field(indexes, 1);
          }

          CAMLreturnT(T, tensor.item<T>());)
}

double at_double_value_at_indexes(value t, value indexes) {
  return at_value_at_indexes<double>(t, indexes);
}

int64_t at_int64_value_at_indexes(value t, value indexes) {
  return at_value_at_indexes<int64_t>(t, indexes);
}

template <typename T> void at_set_value_at_indexes(value t, value indexes, T v) {
  CAMLparam2(t, indexes);
  PROTECT(torch::Tensor tensor = rc_tensor_from_ocaml(t);

          while (indexes != Val_emptylist) {
            tensor = tensor[Int_val(Field(indexes, 0))];
            indexes = Field(indexes, 1);
          }

          tensor.fill_(v);

          CAMLreturn0;)
}

void at_set_double_value_at_indexes(value t, value indexes, double v) {
  at_set_value_at_indexes<double>(t, indexes, v);
}

void at_set_int64_value_at_indexes(value t, value indexes, int64_t v) {
  at_set_value_at_indexes<int64_t>(t, indexes, v);
}

void at_fill_double(value t, double v) {
  CAMLparam1(t);
  PROTECT(rc_tensor_from_ocaml(t).fill_(v);

          CAMLreturn0;)
}

void at_fill_int64(value t, int64_t v) {
  CAMLparam1(t);
  PROTECT(rc_tensor_from_ocaml(t).fill_(v);

          CAMLreturn0;)
}

void at_print(value tensor) {
  CAMLparam1(tensor);
  PROTECT(cout << rc_tensor_from_ocaml(tensor) << endl;

          CAMLreturn0;)
}

value at_to_string(value t, int line_size) {
  CAMLparam1(t);
  PROTECT(std::ostringstream oss;

          torch::print(oss, rc_tensor_from_ocaml(t), line_size);
          CAMLreturn(caml_copy_string(oss.str().c_str()));)
}

void at_copy_(value dst, value src, int non_blocking) {
  CAMLparam2(dst, src);
  PROTECT(rc_tensor_from_ocaml(dst).copy_(rc_tensor_from_ocaml(src), non_blocking);

          CAMLreturn0;)
}
void at_set_data(value dst, value src) {
  CAMLparam2(dst, src);
  PROTECT(rc_tensor_from_ocaml(dst).set_data(rc_tensor_from_ocaml(src));

          CAMLreturn0;)
}

void at_save(value t, const char *filename) {
  CAMLparam1(t);
  PROTECT(torch::save(rc_tensor_from_ocaml(t), filename);

          CAMLreturn0;)
}

void at_save_multi(value tensors, value tensor_names, const char *filename) {
  CAMLparam2(tensors, tensor_names);
  PROTECT(torch::serialize::OutputArchive archive;
          vector<torch::Tensor> tensor_vec = of_ocaml_tensor_list(tensors);
          vector<c10::string_view> tensor_names_vec =
              vec_of_ocaml_string_list(tensor_names);
          for (size_t i = 0; i < tensor_vec.size(); ++i)
              archive.write(std::string(tensor_names_vec[i]), tensor_vec[i],
                            /* buffer=*/false);
          archive.save_to(filename);

          CAMLreturn0;)
}

value at_load_multi(value tensor_names, const char *filename) {
  CAMLparam1(tensor_names);
  PROTECT(torch::serialize::InputArchive archive;
          vector<c10::string_view> tensor_names_vec =
              vec_of_ocaml_string_list(tensor_names);
          archive.load_from(std::string(filename));
          vector<torch::Tensor> ts(tensor_names_vec.size());
          for (size_t i = 0; i < tensor_names_vec.size(); ++i)
              archive.read(std::string(tensor_names_vec[i]), ts[i]);
          // Only allocate the new tensors now so that if there is an exception
          // raised during [read], no memory has to be freed.
          CAMLreturn(to_ocaml_tensor_list(ts));)
}

value at_load_all(const char *filename) {
  CAMLparam0();
  CAMLlocal4(result, prev_cell, curr_cell, curr_tuple);
  PROTECT(auto module = torch::jit::load(filename);

          result = Val_emptylist;

          for (const auto &p
               : module.named_parameters()) {
            curr_cell = caml_alloc_small(2, 0);
            curr_tuple = caml_alloc_small(2, 0);
            Field(curr_tuple, 0) = caml_copy_string(p.name.c_str());
            Field(curr_tuple, 1) = rc_tensor_to_ocaml(p.value);
            Field(curr_cell, 0) = curr_tuple;
            Field(curr_cell, 1) = Val_emptylist;
            if (prev_cell == Val_emptylist) {
              result = curr_cell;
            } else {
              Field(prev_cell, 1) = curr_cell;
            }
            prev_cell = curr_cell;
          }

          CAMLreturn(result);)
}

void at_load_multi_(value tensors, value tensor_names, const char *filename) {
  CAMLparam2(tensors, tensor_names);
  PROTECT(torch::NoGradGuard no_grad;

          torch::serialize::InputArchive archive;

          archive.load_from(std::string(filename));
          vector<torch::Tensor> tensor_vec = of_ocaml_tensor_list(tensors);
          vector<c10::string_view> tensor_names_vec =
              vec_of_ocaml_string_list(tensor_names);
          for (size_t i = 0; i < tensor_names_vec.size(); ++i) {
            torch::Tensor &tensor = tensor_vec[i];
            if (tensor.device().type() == at::kCPU)
              archive.read(std::string(tensor_names_vec[i]), tensor);
            else {
              torch::Tensor tmp_tensor = torch::empty_like(tensor, at::device(at::kCPU));
              archive.read(std::string(tensor_names_vec[i]), tmp_tensor);
              tensor.copy_(tmp_tensor);
            }
          }

          CAMLreturn0;)
}

value at_load(const char *filename) {
  CAMLparam0();
  PROTECT(torch::Tensor tensor;

          torch::load(tensor, filename);

          CAMLreturn(rc_tensor_to_ocaml(tensor));)
}

int at_get_num_interop_threads() { PROTECT(return at::get_num_interop_threads();) }

int at_get_num_threads() { PROTECT(return at::get_num_threads();) }

void at_set_num_interop_threads(int n_threads) {
  PROTECT(at::set_num_interop_threads(n_threads);)
}

void at_set_num_threads(int n_threads) { PROTECT(at::set_num_threads(n_threads);) }

value at_run_backward(value tensor_values, value input_values, int keep_graph,
                      int create_graph) {
  CAMLparam2(tensor_values, input_values);
  PROTECT(
      vector<torch::Tensor> tensors = of_ocaml_tensor_list(tensor_values);
      vector<torch::autograd::Edge> roots;
      for (torch::Tensor &tensor
           : tensors) roots.push_back(torch::autograd::impl::gradient_edge(tensor));

      vector<torch::Tensor> inputs = of_ocaml_tensor_list(input_values);
      vector<torch::autograd::Edge> inputs_;

      for (torch::Tensor &input_
           : inputs) {
        if (!input_.requires_grad())
          throw std::invalid_argument(
              "one of the input tensor does not use set_requires_grad");
        inputs_.push_back(torch::autograd::impl::gradient_edge(input_));
      }

      vector<torch::autograd::Variable>
          grads;
      for (const torch::Tensor &tensor
           : tensors) grads.push_back(torch::ones_like(tensor));

      caml_release_runtime_system();

      torch::autograd::variable_list vl;

      try {
        vl = torch::autograd::Engine::get_default_engine().execute(
            roots, grads, keep_graph, create_graph, false, inputs_);
      } catch (const exception &) {
        caml_acquire_runtime_system();
        throw;
      } caml_acquire_runtime_system();
      CAMLreturn(to_ocaml_tensor_list(vl));)
}

value optim_to_custom_val(torch::optim::Optimizer *ptr) {
  // We create optimizers with a few different derived classes. And unfortunately
  // [pointer_to_custom_val] just casts them all to void pointers. Going through
  // [optim_to_custom_val] ensures that they can all be casted to
  // [torch::optim::Optimizer].
  return pointer_to_custom_val(ptr);
}

value ato_adam(double learning_rate, double beta1, double beta2, double weight_decay,
               double eps) {
  CAMLparam0();
  PROTECT(auto options = torch::optim::AdamOptions(learning_rate)
                             .betas(std::tuple<double, double>(beta1, beta2))
                             .weight_decay(weight_decay)
                             .eps(eps);
          CAMLreturn(optim_to_custom_val(
              new torch::optim::Adam(vector<torch::Tensor>(), options)));)
}

value ato_rmsprop(double learning_rate, double alpha, double eps, double weight_decay,
                  double momentum, int centered) {
  CAMLparam0();
  PROTECT(auto options = torch::optim::RMSpropOptions(learning_rate)
                             .alpha(alpha)
                             .eps(eps)
                             .weight_decay(weight_decay)
                             .momentum(momentum)
                             .centered(centered != 0);
          CAMLreturn(optim_to_custom_val(
              new torch::optim::RMSprop(vector<torch::Tensor>(), options)));)
}

value ato_sgd(double learning_rate, double momentum, double dampening,
              double weight_decay, int nesterov) {
  CAMLparam0();
  PROTECT(auto options = torch::optim::SGDOptions(learning_rate)
                             .momentum(momentum)
                             .dampening(dampening)
                             .weight_decay(weight_decay)
                             .nesterov(nesterov);
          CAMLreturn(optim_to_custom_val(
              new torch::optim::SGD(vector<torch::Tensor>(), options)));)
}

void ato_add_parameters(value t_in, value tensors_list) {
  CAMLparam2(t_in, tensors_list);
  PROTECT(vector<torch::Tensor> tensors = of_ocaml_tensor_list(tensors_list);
          optimizer t = pointer_of_custom_val<torch::optim::Optimizer>(t_in);

          for (torch::Tensor &tensor
               : tensors) t->param_groups()[0]
              .params()
              .push_back(std::move(tensor));

          CAMLreturn0;)
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

void ato_set_learning_rate(value t_in, double learning_rate) {
  CAMLparam1(t_in);
  PROTECT(optimizer t = pointer_of_custom_val<torch::optim::Optimizer>(t_in);
          set_lr<torch::optim::AdamOptions>(t, learning_rate);
          set_lr<torch::optim::AdamWOptions>(t, learning_rate);
          set_lr<torch::optim::RMSpropOptions>(t, learning_rate);
          set_lr<torch::optim::SGDOptions>(t, learning_rate);

          CAMLreturn0;)
}

template <class T> void set_lr_group(optimizer t, size_t group, double learning_rate) {
  auto &param_group = t->param_groups().at(group);
  torch::optim::OptimizerOptions *d = &(param_group.options());
  if (auto p = dynamic_cast<T *>(d)) {
    p->lr(learning_rate);
  }
}

void ato_set_learning_rate_group(value t_in, size_t group, double learning_rate) {
  CAMLparam1(t_in);
  PROTECT(optimizer t = pointer_of_custom_val<torch::optim::Optimizer>(t_in);
          set_lr_group<torch::optim::AdamOptions>(t, group, learning_rate);
          set_lr_group<torch::optim::AdamWOptions>(t, group, learning_rate);
          set_lr_group<torch::optim::RMSpropOptions>(t, group, learning_rate);
          set_lr_group<torch::optim::SGDOptions>(t, group, learning_rate);

          CAMLreturn0;)
}

void ato_set_momentum(value t_in, double momentum) {
  CAMLparam1(t_in);
  PROTECT(
      optimizer t = pointer_of_custom_val<torch::optim::Optimizer>(t_in);
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
      } else throw std::invalid_argument("unexpected optimizer");

      CAMLreturn0;)
}

void ato_set_momentum_group(value t_in, size_t group, double momentum) {
  CAMLparam1(t_in);
  PROTECT(
      optimizer t = pointer_of_custom_val<torch::optim::Optimizer>(t_in);
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
      } else throw std::invalid_argument("unexpected optimizer");

      CAMLreturn0;)
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

void ato_set_weight_decay(value t_in, double weight_decay) {
  CAMLparam1(t_in);
  PROTECT(optimizer t = pointer_of_custom_val<torch::optim::Optimizer>(t_in);
          set_weight_decay<torch::optim::AdamOptions>(t, weight_decay);
          set_weight_decay<torch::optim::AdamWOptions>(t, weight_decay);
          set_weight_decay<torch::optim::RMSpropOptions>(t, weight_decay);
          set_weight_decay<torch::optim::SGDOptions>(t, weight_decay);

          CAMLreturn0;)
}

template <class T>
void set_weight_decay_group(optimizer t, size_t group, double weight_decay) {
  auto &param_group = t->param_groups().at(group);
  torch::optim::OptimizerOptions *d = &(param_group.options());
  if (auto p = dynamic_cast<T *>(d)) {
    p->weight_decay(weight_decay);
  }
}

void ato_set_weight_decay_group(value t_in, size_t group, double weight_decay) {
  CAMLparam1(t_in);
  PROTECT(optimizer t = pointer_of_custom_val<torch::optim::Optimizer>(t_in);
          set_weight_decay_group<torch::optim::AdamOptions>(t, group, weight_decay);
          set_weight_decay_group<torch::optim::AdamWOptions>(t, group, weight_decay);
          set_weight_decay_group<torch::optim::RMSpropOptions>(t, group, weight_decay);
          set_weight_decay_group<torch::optim::SGDOptions>(t, group, weight_decay);

          CAMLreturn0;)
}

void ato_zero_grad(value t) {
  CAMLparam1(t);
  PROTECT(pointer_of_custom_val<torch::optim::Optimizer>(t)->zero_grad();

          CAMLreturn0;)
}

void ato_step(value t) {
  CAMLparam1(t);
  PROTECT(pointer_of_custom_val<torch::optim::Optimizer>(t)->step();

          CAMLreturn0;)
}

void ato_free(value t) {
  CAMLparam1(t);
  delete pointer_of_custom_val<torch::optim::Optimizer>(t);
  CAMLreturn0;
}

value ats_int(int64_t v) {
  CAMLparam0();
  PROTECT(CAMLreturn(pointer_to_custom_val(new torch::Scalar(v)));)
}

value ats_float(double v) {
  CAMLparam0();
  PROTECT(CAMLreturn(pointer_to_custom_val(new torch::Scalar(v)));)
}

int64_t ats_to_int(value s) { PROTECT(return scalar_from_ocaml_noalloc(s)->toLong();) }

double ats_to_float(value s) { PROTECT(return scalar_from_ocaml_noalloc(s)->toDouble();) }

void ats_free(value s) {
  CAMLparam1(s);
  delete scalar_from_ocaml_noalloc(s);
  CAMLreturn0;
}

int atc_cuda_device_count() { PROTECT(return torch::cuda::device_count();) }

int atc_cuda_is_available() { PROTECT(return torch::cuda::is_available();) }

int atc_cudnn_is_available() { PROTECT(return torch::cuda::cudnn_is_available();) }

void atc_set_benchmark_cudnn(int b) { at::globalContext().setBenchmarkCuDNN(b); }

class no_runtime_system {
public:
  no_runtime_system() { caml_release_runtime_system(); }
  ~no_runtime_system() { caml_acquire_runtime_system(); }
};

value aoti_runner_cuda_load(const char *filename_cstr, int num_concurrent_executions,
                            int device, const char *cubin_dir_cstr) {
  CAMLparam0();
  PROTECT({
    std::string filename = filename_cstr;
    std::string cubin_dir = cubin_dir_cstr;
    torch::inductor::AOTIModelContainerRunnerCuda *runner = nullptr;
    // Release the runtime lock because loading the weights can be slow. But
    // first make copies of the strings, so it's safe to use them without the
    // lock. We must not allocate in this section, so [pointer_to_custom_val]
    // has to be called outside of it.
    {
      no_runtime_system nosys;
      at::Device torch_device = device_of_int(device);
      runner = new torch::inductor::AOTIModelContainerRunnerCuda(
          filename, num_concurrent_executions, torch_device.str(), cubin_dir,
          true /* run_single_threaded */);
    }
    CAMLreturn(pointer_to_custom_val(runner));
  })
}

void aoti_runner_cuda_run_unit(value r, value inputs_list) {
  CAMLparam2(r, inputs_list);
  PROTECT(
      std::vector<torch::Tensor> inputs = of_ocaml_tensor_list(inputs_list);
      // release the runtime lock because running the aoti kernel could take several ms
      {
        no_runtime_system nosys;
        pointer_of_custom_val<torch::inductor::AOTIModelContainerRunnerCuda>(r)->run(
            inputs);
      }
      //
      CAMLreturn0;)
}

void aoti_runner_cuda_free(value r) {
  CAMLparam1(r);
  delete pointer_of_custom_val<torch::inductor::AOTIModelContainerRunnerCuda>(r);
  CAMLreturn0;
}

void at_set_graph_executor_optimize(bool o) { torch::jit::setGraphExecutorOptimize(o); }

value atg_zeros_rc(value size_list, int kind, int device) {
  CAMLparam1(size_list);
  PROTECT(std::vector<int64_t> size_vec = vec_of_ocaml_int_list(size_list);
          torch::Tensor result = torch::zeros(
              size_vec, at::device(device_of_int(device)).dtype(at::ScalarType(kind)));
          CAMLreturn(rc_tensor_to_ocaml(result));)
}

void torch_record_memory_history() { torch::cuda::_record_memory_history(); }

void torch_save_memory_snapshot_pickled(const char *output_filepath) {
  auto snapshot_data = torch::cuda::_memory_snapshot_pickled();
  PROTECT(std::ofstream snapshot_file(output_filepath, std::ios_base::binary);
          snapshot_file.write(snapshot_data.data(), snapshot_data.size());)
}

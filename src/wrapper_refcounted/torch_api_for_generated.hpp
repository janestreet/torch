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
#include "ocaml_runtime_props.h"
#include "ocaml_to_cpp.h"

using namespace std;

torch::Tensor rc_tensor_from_ocaml(value tensor);
value rc_tensor_to_ocaml(const torch::Tensor &tensor);
std::optional<torch::Tensor> tensor_option_from_ocaml(value tensor_option);
value to_ocaml_tensor_list(const std::vector<torch::Tensor> &tensors);
std::vector<torch::Tensor> of_ocaml_tensor_list(value tensor_list);

torch::Scalar *scalar_from_ocaml_noalloc(value scalar);
std::optional<torch::Scalar> scalar_option_from_ocaml(value scalar_opt);

at::Device device_of_int(int d);
c10::optional<at::Device> optional_device_of_int(int d);

template <size_t I, typename... Ts>
void one_rc_tensor_to_ocaml(value tuple, const std::tuple<Ts...> &tensors) {
  CAMLparam1(tuple);
  // Turning a C++ tuple into an ocaml tuple is a bit crazy. You need to iterate over the
  // tuple at compile-time but there is no language feature to iterate at compile time. So
  // instead you need helper functions like this one, which just does one element.
  Store_field(tuple, I, rc_tensor_to_ocaml(std::get<I>(tensors)));
  CAMLreturn0;
}

template <typename... Ts, size_t... Is>
value rc_tensors_to_ocaml_tuple_index_sequence(const std::tuple<Ts...> &tensors,
                                               std::index_sequence<Is...>) {
  CAMLparam0();
  CAMLlocal1(tensors_tuple);
  tensors_tuple = caml_alloc_tuple(sizeof...(Is));
  // Call the above function with every element in the index sequence. The ", ..." will
  // repeat what comes before for every element in Is. (The placement of "..." relative to
  // "Is" indicates what should be repeated.)
  (one_rc_tensor_to_ocaml<Is>(tensors_tuple, tensors), ...);
  CAMLreturn(tensors_tuple);
}

template <typename... Ts>
value rc_tensors_to_ocaml_tuple(const std::tuple<Ts...> &tensors) {
  CAMLparam0();
  // In order to allow iterating in the functions above we need a compile-time list of
  // indices. The only way to create that is with another level of indirection so this
  // function just calls the above function with the index_sequence.
  CAMLreturn(rc_tensors_to_ocaml_tuple_index_sequence(
      tensors, std::make_index_sequence<sizeof...(Ts)>()));
}

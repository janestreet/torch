#include <vector>
#include <optional>
#include <caml/custom.h>
#include <torch/torch.h>

size_t list_length_noalloc(value list);
std::optional<double> optional_double_from_ocaml(value float_option);
std::optional<int64_t> optional_int_from_ocaml(value int_option);
c10::optional<c10::string_view> optional_string_from_ocaml(value string_option);
std::vector<int64_t> vec_of_ocaml_int_list(value int_list);
std::vector<double> vec_of_ocaml_double_list(value double_list);
std::vector<c10::string_view> vec_of_ocaml_string_list(value string_list);
value ocaml_list_of_ints(c10::ArrayRef<int64_t> int_list);

value pointer_to_custom_val(void *ptr, size_t size);
template <typename T> value pointer_to_custom_val(T *ptr) {
  return pointer_to_custom_val(static_cast<void *>(ptr), sizeof(T));
}
template <typename T> T *pointer_of_custom_val(value v) {
  return *reinterpret_cast<T **>(Data_custom_val(v));
}

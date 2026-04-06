#include "ocaml_to_cpp.h"
#include <caml/memory.h>
#include <caml/alloc.h>

size_t list_length_noalloc(value list) {
  size_t result = 0;
  while (list != Val_emptylist) {
    ++result;
    list = Field(list, 1);
  }
  return result;
}

std::optional<double> optional_double_from_ocaml(value float_option) {
  CAMLparam1(float_option);
  CAMLlocal1(caml_float);
  std::optional<double> result = std::nullopt;

  if (Is_some(float_option)) {
    caml_float = Some_val(float_option);
    result = std::optional<double>{Double_val(caml_float)};
  }

  CAMLreturnT(std::optional<double>, result);
}

std::optional<int64_t> optional_int_from_ocaml(value int_option) {
  CAMLparam1(int_option);
  CAMLlocal1(caml_int);
  std::optional<int64_t> result = std::nullopt;

  if (Is_some(int_option)) {
    caml_int = Some_val(int_option);
    result = std::optional<int64_t>{Int_val(caml_int)};
  }

  CAMLreturnT(std::optional<int64_t>, result);
}

c10::optional<c10::string_view> optional_string_from_ocaml(value string_option) {
  CAMLparam1(string_option);
  CAMLlocal1(caml_str);
  c10::optional<c10::string_view> result = c10::nullopt;

  if (Is_some(string_option)) {
    caml_str = Some_val(string_option);
    result = c10::optional<c10::string_view>(String_val(caml_str));
  }

  CAMLreturnT(c10::optional<c10::string_view>, result);
}

std::vector<int64_t> vec_of_ocaml_int_list(value int_list) {
  CAMLparam1(int_list);
  CAMLlocal1(list_ptr);
  list_ptr = int_list;
  std::vector<int64_t> result;
  result.reserve(list_length_noalloc(list_ptr));

  while (list_ptr != Val_emptylist) {
    result.push_back(Int_val(Field(list_ptr, 0)));
    list_ptr = Field(list_ptr, 1);
  }

  CAMLreturnT(std::vector<int64_t>, result);
}

std::vector<double> vec_of_ocaml_double_list(value double_list) {
  CAMLparam1(double_list);
  CAMLlocal1(list_ptr);
  list_ptr = double_list;
  std::vector<double> result;
  result.reserve(list_length_noalloc(list_ptr));

  while (list_ptr != Val_emptylist) {
    result.push_back(Double_val(Field(list_ptr, 0)));
    list_ptr = Field(list_ptr, 1);
  }

  CAMLreturnT(std::vector<double>, result);
}

std::vector<c10::string_view> vec_of_ocaml_string_list(value string_list) {
  CAMLparam1(string_list);
  CAMLlocal1(list_ptr);
  list_ptr = string_list;
  std::vector<c10::string_view> result;
  result.reserve(list_length_noalloc(list_ptr));

  while (list_ptr != Val_emptylist) {
    result.emplace_back(String_val(Field(list_ptr, 0)));
    list_ptr = Field(list_ptr, 1);
  }

  CAMLreturnT(std::vector<c10::string_view>, result);
}

value ocaml_list_of_ints(c10::ArrayRef<int64_t> int_list) {
  CAMLparam0();
  CAMLlocal2(next_cell, curr_cell);
  next_cell = Val_emptylist;

  for (auto it = int_list.rbegin(); it != int_list.rend(); ++it) {
    curr_cell = caml_alloc_small(2, 0);
    Field(curr_cell, 0) = Val_int(*it);
    Field(curr_cell, 1) = next_cell;
    next_cell = curr_cell;
  }

  CAMLreturn(next_cell);
}

// Since we have no custom behavior, we can use the same ops for all types of objects.
// Tensors, scalars, optimizers, they all use the same ops. We could possibly have
// different finalizers here for different objects, but we can also just add those in
// OCaml. Especially for tensors where we only attach a finalizer when switching from RC
// to GC.
static struct custom_operations default_ops = {"torch-object",
                                               custom_finalize_default,
                                               custom_compare_default,
                                               custom_hash_default,
                                               custom_serialize_default,
                                               custom_deserialize_default,
                                               custom_compare_ext_default,
                                               custom_fixed_length_default};

value pointer_to_custom_val(void *ptr, size_t size) {
  CAMLparam0();
  CAMLlocal1(caml_value);
  caml_value = caml_alloc_custom_mem(&default_ops, sizeof(ptr), size);
  *reinterpret_cast<void **>(Data_custom_val(caml_value)) = ptr;
  CAMLreturn(caml_value);
}

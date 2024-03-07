#include "torch_api.h"
#include "ctypes_cstubs_internals.h"
#include <caml/memory.h>

// returns a voidp (ocaml block storing a void pointer)
CAMLprim value with_tensor_gc(value raw_tensor_value) {
  CAMLparam1(raw_tensor_value);
  raw_tensor raw = CTYPES_ADDR_OF_FATPTR(raw_tensor_value);
  CAMLreturn(with_tensor_gc_internal(raw));
}

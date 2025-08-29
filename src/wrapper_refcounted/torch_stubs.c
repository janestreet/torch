#include "torch_api.h"
#include "ctypes_cstubs_internals.h"
#include <caml/memory.h>

CAMLprim value make_managed_tensor(value addr) {
  CAMLparam1(addr);
  CAMLreturn(make_managed_tensor_internal(addr));
}

CAMLprim value increment_refcount(value gc_tensor) {
  CAMLparam1(gc_tensor);
  increment_refcount_internal(gc_tensor);
  CAMLreturn(Val_unit);
}

CAMLprim value decrement_refcount(value gc_tensor) {
  CAMLparam1(gc_tensor);
  decrement_refcount_internal(gc_tensor);
  CAMLreturn(Val_unit);
}

CAMLprim value get_refcount(value gc_tensor) {
  CAMLparam1(gc_tensor);
  int count = get_refcount_internal(gc_tensor);
  CAMLreturn(Val_int(count));
}

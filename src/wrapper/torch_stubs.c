#include "torch_api.h"
#include "ctypes_cstubs_internals.h"
#include <caml/memory.h>

CAMLprim value make_managed_tensor(value addr) {
  CAMLparam1(addr);
  CAMLreturn(make_managed_tensor_internal(addr));
}

CAMLprim value finalize_managed_tensor(value managed) {
  CAMLparam1(managed);
  finalize_managed_tensor_internal(managed);
  CAMLreturn(Val_unit);
}

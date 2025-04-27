#pragma once

/* This function returns a [Torch_core.C_ffi.unwrapped_managed_tensor]. You need to call
   [Torch_core.C_ffi.wrap_managed_tensor] on it to get a real tensor. */
value prepare_ocaml_tensor(const torch::Tensor &tensor);

#pragma once

extern "C" void torch_refcounted_check_abi_once();

/* This function returns a [Torch_core.C_ffi.unwrapped_managed_tensor]. You need to call
   [Torch_core.C_ffi.wrap_managed_tensor] on it to get a real tensor. */
extern "C" value prepare_ocaml_tensor(const torch::Tensor &tensor);

module C_ffi = C_ffi
module Device = Torch_wrapper_types.Device
module Kind = Torch_wrapper_types.Kind
module Reduction = Torch_wrapper_types.Reduction
module Wrapper = Wrapper

module Wrapper_generated = struct
  include Wrapper_generated_refcounted0
  include Wrapper_generated_refcounted1
  include Wrapper_generated_refcounted2
  include Wrapper_generated_refcounted3
  include Wrapper_generated_refcounted4
  include Wrapper_generated_refcounted5
  include Wrapper_generated_refcounted6
  include Wrapper_generated_refcounted7
end

module Wrapper_generated_intf = Wrapper_generated_refcounted_intf

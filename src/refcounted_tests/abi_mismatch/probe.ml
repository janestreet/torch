let () =
  Core_unix.RLimit.(set core_file_size { cur = Limit 0L; max = Limit 0L });
  Torch_refcounted_core.Wrapper.manual_seed 0
;;

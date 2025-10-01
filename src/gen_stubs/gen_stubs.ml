let run ~refcounted =
  let fmt file = Format.formatter_of_out_channel (open_out file) in
  let filename_base =
    if refcounted then "torch_refcounted_stubs_generated" else "torch_stubs_generated"
  in
  let stubs_c_filename = filename_base ^ ".c" in
  let stubs_ml_filename = filename_base ^ ".ml" in
  let fmt_c = fmt stubs_c_filename in
  Format.fprintf fmt_c "#include \"torch_api.h\"@.";
  Cstubs.write_c
    fmt_c
    ~prefix:"caml_"
    (if refcounted
     then (module Torch_refcounted_bindings.C)
     else (module Torch_bindings.C));
  let fmt_ml = fmt stubs_ml_filename in
  Cstubs.write_ml
    fmt_ml
    ~prefix:"caml_"
    (if refcounted
     then (module Torch_refcounted_bindings.C)
     else (module Torch_bindings.C));
  flush_all ()
;;

let command =
  Command.basic
    ~summary:"generate stubs for torch functions"
    (let%map_open.Command refcounted =
       flag
         "refcounted"
         (optional_with_default false bool)
         ~doc:
           "BOOL if set, generated stubs will use the refcounted bindings module and \
            output files will be named to indicate they are for the refcounted \
            implementation"
     in
     fun () -> run ~refcounted)
;;

let () = Command_unix.run command

let run () =
  let fmt file = Format.formatter_of_out_channel (open_out file) in
  let filename_base = "torch_stubs_generated" in
  let stubs_c_filename = filename_base ^ ".c" in
  let stubs_ml_filename = filename_base ^ ".ml" in
  let fmt_c = fmt stubs_c_filename in
  Format.fprintf fmt_c "#include \"torch_api.h\"@.";
  Cstubs.write_c fmt_c ~prefix:"caml_" (module Torch_bindings.C);
  let fmt_ml = fmt stubs_ml_filename in
  Cstubs.write_ml fmt_ml ~prefix:"caml_" (module Torch_bindings.C);
  flush_all ()
;;

let command =
  Command.basic
    ~summary:"generate stubs for torch functions"
    (let%map_open.Command () = return () in
     fun () -> run ())
;;

let () = Command_unix.run command

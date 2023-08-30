open Base
module C = Configurator.V1

let empty_flags = { C.Pkg_config.cflags = []; libs = [] }

let combine (flags1 : C.Pkg_config.package_conf) (flags2 : C.Pkg_config.package_conf) =
  { C.Pkg_config.cflags = flags1.cflags @ flags2.cflags
  ; libs = flags1.libs @ flags2.libs
  }
;;

let ( /^ ) = Stdlib.Filename.concat
let file_exists = Stdlib.Sys.file_exists

let extract_flags c ~package =
  Option.bind (C.Pkg_config.get c) ~f:(C.Pkg_config.query ~package)
;;

let or_else o ~f =
  match o with
  | Some _ -> o
  | None -> f ()
;;

let dynamic_links = [ "-lc10"; "-ltorch_cpu"; "-ltorch" ]

let torch_flags () =
  let config ~include_dir ~lib_dir =
    let cflags =
      [ "-isystem"
      ; Printf.sprintf "%s" include_dir
      ; "-isystem"
      ; Printf.sprintf "%s/torch/csrc/api/include" include_dir
      ]
    in
    let libs =
      [ Printf.sprintf "-Wl,-rpath,%s" lib_dir; Printf.sprintf "-L%s" lib_dir ]
      @ dynamic_links
    in
    { C.Pkg_config.cflags; libs }
  in
  let conda_config ~conda_prefix =
    let conda_prefix = conda_prefix ^ "/lib" in
    Stdlib.Sys.readdir conda_prefix
    |> Array.to_list
    |> List.filter_map ~f:(fun filename ->
         if String.is_prefix filename ~prefix:"python"
         then (
           let libdir =
             Printf.sprintf "%s/%s/site-packages/torch" conda_prefix filename
           in
           if file_exists libdir && Stdlib.Sys.is_directory libdir
           then Some libdir
           else None)
         else None)
    |> function
    | [] -> None
    | lib_dir :: _ ->
      Some (config ~include_dir:(lib_dir /^ "include") ~lib_dir:(lib_dir /^ "lib"))
  in
  let flags =
    (* try libtorch env var *)
    Option.map (Stdlib.Sys.getenv_opt "LIBTORCH") ~f:(fun l ->
      config ~include_dir:(l /^ "include") ~lib_dir:(l /^ "lib"))
    (* try system libraries *)
    |> or_else ~f:(fun () ->
         match Stdlib.Sys.getenv_opt "LIBTORCH_USE_SYSTEM" with
         | Some "1" -> Some { C.Pkg_config.cflags = []; libs = dynamic_links }
         | _ -> None)
    (* try conda environment *)
    |> or_else ~f:(fun () ->
         Option.bind (Stdlib.Sys.getenv_opt "CONDA_PREFIX") ~f:(fun conda_prefix ->
           conda_config ~conda_prefix))
    (* try opam switch *)
    |> or_else ~f:(fun () ->
         Option.bind (Stdlib.Sys.getenv_opt "OPAM_SWITCH_PREFIX") ~f:(fun prefix ->
           let lib_dir = prefix /^ "lib" /^ "libtorch" in
           if file_exists lib_dir
           then
             Some (config ~include_dir:(lib_dir ^ "/include") ~lib_dir:(lib_dir ^ "/lib"))
           else None))
  in
  Option.value flags ~default:empty_flags
;;

let libcuda_flags ~lcuda ~lnvrtc =
  let cudadir = "/usr/local/cuda/lib64" in
  if file_exists cudadir && Stdlib.Sys.is_directory cudadir
  then (
    let libs =
      [ Printf.sprintf "-Wl,-rpath,%s" cudadir; Printf.sprintf "-L%s" cudadir ]
    in
    let libs = if lcuda then libs @ [ "-lcudart" ] else libs in
    let libs = if lnvrtc then libs @ [ "-lnvrtc" ] else libs in
    { C.Pkg_config.cflags = []; libs })
  else empty_flags
;;

let () =
  C.main ~name:"torch-config" (fun c ->
    let torch_flags =
      try torch_flags () with
      | _ -> empty_flags
    in
    let cuda_flags = extract_flags c ~package:"cuda" in
    let nvrtc_flags = extract_flags c ~package:"nvrtc" in
    let cuda_flags =
      match cuda_flags, nvrtc_flags with
      | None, None -> libcuda_flags ~lcuda:true ~lnvrtc:true
      | Some cuda_flags, None ->
        combine cuda_flags (libcuda_flags ~lcuda:false ~lnvrtc:true)
      | None, Some nvrtc_flags ->
        combine nvrtc_flags (libcuda_flags ~lcuda:true ~lnvrtc:false)
      | Some cuda_flags, Some nvrtc_flags -> combine cuda_flags nvrtc_flags
    in
    let conda_libs =
      Option.value_map
        (Stdlib.Sys.getenv_opt "CONDA_PREFIX")
        ~f:(fun conda_prefix -> [ Printf.sprintf "-Wl,-rpath,%s/lib" conda_prefix ])
        ~default:[]
    in
    let cxx_abi_flag =
      let cxx_abi =
        match Stdlib.Sys.getenv_opt "LIBTORCH_CXX11_ABI" with
        | Some v -> v
        | None -> "1"
      in
      Printf.sprintf "-D_GLIBCXX_USE_CXX11_ABI=%s" cxx_abi
    in
    C.Flags.write_sexp
      "cxx_flags.sexp"
      (cxx_abi_flag :: (torch_flags.cflags @ cuda_flags.cflags));
    let torch_flags_lib =
      if Stdlib.( = ) cuda_flags empty_flags
      then torch_flags.libs
      else "-Wl,--no-as-needed" :: torch_flags.libs
    in
    C.Flags.write_sexp
      "c_library_flags.sexp"
      (torch_flags_lib @ conda_libs @ cuda_flags.libs))
;;

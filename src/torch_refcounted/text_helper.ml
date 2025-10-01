open Base

type t =
  { content : Tensor.t
  ; char_for_label : char Map.M(Int).t
  }

let create ~filename =
  let file_descr = Unix.openfile filename [ O_RDONLY ] 0 in
  let content =
    Unix.map_file file_descr Int8_unsigned C_layout false [| -1 |]
    |> Bigarray.array1_of_genarray
  in
  Unix.close file_descr;
  let label_for_char = Hashtbl.Poly.create () in
  for i = 0 to Bigarray.Array1.dim content - 1 do
    content.{i}
    <- Hashtbl.find_or_add label_for_char content.{i} ~default:(fun () ->
         Hashtbl.length label_for_char)
  done;
  let char_for_label =
    Hashtbl.to_alist label_for_char
    |> List.map ~f:(fun (char, label) -> label, Char.of_int_exn char)
    |> Map.of_alist_exn (module Int)
  in
  let content =
    Tensor.with_rc_scope (fun () ->
      let tensor =
        Bigarray.genarray_of_array1 content
        |> Tensor.of_bigarray
        |> Tensor.convert_rc_tensor_to_gc
      in
      tensor)
  in
  { content; char_for_label }
;;

let total_length t = Tensor.shape t.content |> List.hd_exn
let char t ~label = Map.find_exn t.char_for_label label
let labels t = Map.length t.char_for_label

let iter ?device t ~f ~seq_len ~batch_size =
  Tensor.with_rc_scope (fun () ->
    let total_length = total_length t in
    let start_indexes =
      Tensor.randperm ~n:(total_length - seq_len) ~options:(T Int64, Cpu)
    in
    for index = 0 to ((total_length - seq_len - 1) / batch_size) - 1 do
      Tensor.with_rc_scope (fun () ->
        let xs, ys =
          Torch_local_iterators.List.init_local batch_size ~f:(fun i -> exclave_
            let start =
              Tensor.get start_indexes ((index * batch_size) + i) |> Tensor.int_value
            in
            ( Tensor.narrow t.content ~dim:0 ~start ~length:seq_len
            , Tensor.narrow t.content ~dim:0 ~start:(start + 1) ~length:seq_len ))
          |> Torch_local_iterators.List.unzip_local
        in
        let stack v = exclave_
          Tensor.stack v ~dim:0
          |> Tensor.to_device ?device
          |> Tensor.to_type ~type_:(T Int64)
        in
        f index ~xs:(stack xs) ~ys:(stack ys);
        ())
    done)
;;

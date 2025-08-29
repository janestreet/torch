type t =
  | Cpu
  | Cuda of int

(* Hardcoded, should match torch_api.cpp *)
let option_to_int = function
  | None -> -2
  | Some Cpu -> -1
  | Some (Cuda i) ->
    if i < 0 then Printf.sprintf "negative index for cuda device" |> failwith;
    i
;;

let to_int t = option_to_int (Some t)
let of_int i = if i < 0 then Cpu else Cuda i

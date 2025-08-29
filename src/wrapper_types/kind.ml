type _ t =
  | Uint8 : [ `u8 ] t
  | Int8 : [ `i8 ] t
  | Int16 : [ `i16 ] t
  | Int : [ `i32 ] t
  | Int64 : [ `i64 ] t
  | Half : [ `f16 ] t
  | Float : [ `f32 ] t
  | Double : [ `f64 ] t
  | ComplexHalf : [ `c16 ] t
  | ComplexFloat : [ `c32 ] t
  | ComplexDouble : [ `c64 ] t
  | Bool : [ `bool ] t
  | QInt8 : [ `qi8 ] t
  | QUInt8 : [ `qu8 ] t
  | QInt32 : [ `qi32 ] t
  | BFloat16 : [ `bf16 ] t

let u8 = Uint8
let i8 = Int8
let i16 = Int16
let i32 = Int
let i64 = Int64
let f16 = Half
let f32 = Float
let f64 = Double
let c16 = ComplexHalf
let c32 = ComplexFloat
let c64 = ComplexDouble
let bool = Bool

type packed = T : _ t -> packed

(* Hardcoded, should match ScalarType.h *)
let to_int : type a. a t -> int = function
  | Uint8 -> 0
  | Int8 -> 1
  | Int16 -> 2
  | Int -> 3
  | Int64 -> 4
  | Half -> 5
  | Float -> 6
  | Double -> 7
  | ComplexHalf -> 8
  | ComplexFloat -> 9
  | ComplexDouble -> 10
  | Bool -> 11
  | QInt8 -> 12
  | QUInt8 -> 13
  | QInt32 -> 14
  | BFloat16 -> 15
;;

let packed_to_int (T t) = to_int t

let of_int_exn = function
  | 0 -> T Uint8
  | 1 -> T Int8
  | 2 -> T Int16
  | 3 -> T Int
  | 4 -> T Int64
  | 5 -> T Half
  | 6 -> T Float
  | 7 -> T Double
  | 8 -> T ComplexHalf
  | 9 -> T ComplexFloat
  | 10 -> T ComplexDouble
  | 11 -> T Bool
  | 12 -> T QInt8
  | 13 -> T QUInt8
  | 14 -> T QInt32
  | 15 -> T BFloat16
  | d -> failwith (Printf.sprintf "unexpected kind %d" d)
;;

let ( <> ) packed1 packed2 = packed_to_int packed1 <> packed_to_int packed2

let size_in_bytes = function
  | T Uint8 -> 1
  | T Int8 -> 1
  | T Int16 -> 2
  | T Int -> 4
  | T Int64 -> 8
  | T Half -> 2
  | T Float -> 4
  | T Double -> 8
  | T ComplexHalf -> 4
  | T ComplexFloat -> 8
  | T ComplexDouble -> 16
  | T Bool -> 1
  | T QInt8 -> 1
  | T QUInt8 -> 1
  | T QInt32 -> 4
  | T BFloat16 -> 2
;;

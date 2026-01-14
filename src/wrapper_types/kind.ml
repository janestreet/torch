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
  | QUInt4x2 : [ `quint4x2 ] t
  | QUInt2x4 : [ `quint2x4 ] t
  | Bits1x8 : [ `bits1x8 ] t
  | Bits2x4 : [ `bits2x4 ] t
  | Bits4x2 : [ `bits4x2 ] t
  | Bits8 : [ `bits8 ] t
  | Bits16 : [ `bits16 ] t
  | Float8_e5m2 : [ `float8_e5m2 ] t
  | Float8_e4m3fn : [ `float8_e4m3fn ] t
  | Float8_e5m2fnuz : [ `float8_e5m2fnuz ] t
  | Float8_e4m3fnuz : [ `float8_e4m3fnuz ] t
  | UInt16 : [ `uint16 ] t
  | UInt32 : [ `uint32 ] t
  | UInt64 : [ `uint64 ] t
  | UInt1 : [ `uint1 ] t
  | UInt2 : [ `uint2 ] t
  | UInt3 : [ `uint3 ] t
  | UInt4 : [ `uint4 ] t
  | UInt5 : [ `uint5 ] t
  | UInt6 : [ `uint6 ] t
  | UInt7 : [ `uint7 ] t
  | Int1 : [ `int1 ] t
  | Int2 : [ `int2 ] t
  | Int3 : [ `int3 ] t
  | Int4 : [ `int4 ] t
  | Int5 : [ `int5 ] t
  | Int6 : [ `int6 ] t
  | Int7 : [ `int7 ] t
  | Float8_e8m0fnu : [ `float8_e8m0fnu ] t
  | Float4_e2m1fn_x2 : [ `float4_e2m1fn_x2 ] t

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
  | QUInt4x2 -> 16
  | QUInt2x4 -> 17
  | Bits1x8 -> 18
  | Bits2x4 -> 19
  | Bits4x2 -> 20
  | Bits8 -> 21
  | Bits16 -> 22
  | Float8_e5m2 -> 23
  | Float8_e4m3fn -> 24
  | Float8_e5m2fnuz -> 25
  | Float8_e4m3fnuz -> 26
  | UInt16 -> 27
  | UInt32 -> 28
  | UInt64 -> 29
  | UInt1 -> 30
  | UInt2 -> 31
  | UInt3 -> 32
  | UInt4 -> 33
  | UInt5 -> 34
  | UInt6 -> 35
  | UInt7 -> 36
  | Int1 -> 37
  | Int2 -> 38
  | Int3 -> 39
  | Int4 -> 40
  | Int5 -> 41
  | Int6 -> 42
  | Int7 -> 43
  | Float8_e8m0fnu -> 44
  | Float4_e2m1fn_x2 -> 45
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
  | 16 -> T QUInt4x2
  | 17 -> T QUInt2x4
  | 18 -> T Bits1x8
  | 19 -> T Bits2x4
  | 20 -> T Bits4x2
  | 21 -> T Bits8
  | 22 -> T Bits16
  | 23 -> T Float8_e5m2
  | 24 -> T Float8_e4m3fn
  | 25 -> T Float8_e5m2fnuz
  | 26 -> T Float8_e4m3fnuz
  | 27 -> T UInt16
  | 28 -> T UInt32
  | 29 -> T UInt64
  | 30 -> T UInt1
  | 31 -> T UInt2
  | 32 -> T UInt3
  | 33 -> T UInt4
  | 34 -> T UInt5
  | 35 -> T UInt6
  | 36 -> T UInt7
  | 37 -> T Int1
  | 38 -> T Int2
  | 39 -> T Int3
  | 40 -> T Int4
  | 41 -> T Int5
  | 42 -> T Int6
  | 43 -> T Int7
  | 44 -> T Float8_e8m0fnu
  | 45 -> T Float4_e2m1fn_x2
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
  | T QUInt4x2 -> 1
  | T QUInt2x4 -> 1
  | T Bits1x8 -> 1
  | T Bits2x4 -> 1
  | T Bits4x2 -> 1
  | T Bits8 -> 1
  | T Bits16 -> 2
  | T Float8_e5m2 -> 1
  | T Float8_e4m3fn -> 1
  | T Float8_e5m2fnuz -> 1
  | T Float8_e4m3fnuz -> 1
  | T UInt16 -> 2
  | T UInt32 -> 4
  | T UInt64 -> 8
  | T UInt1 -> 1
  | T UInt2 -> 1
  | T UInt3 -> 1
  | T UInt4 -> 1
  | T UInt5 -> 1
  | T UInt6 -> 1
  | T UInt7 -> 1
  | T Int1 -> 1
  | T Int2 -> 1
  | T Int3 -> 1
  | T Int4 -> 1
  | T Int5 -> 1
  | T Int6 -> 1
  | T Int7 -> 1
  | T Float8_e8m0fnu -> 1
  | T Float4_e2m1fn_x2 -> 1
;;

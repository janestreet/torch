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

val u8 : [ `u8 ] t
val i8 : [ `i8 ] t
val i16 : [ `i16 ] t
val i32 : [ `i32 ] t
val i64 : [ `i64 ] t
val f16 : [ `f16 ] t
val f32 : [ `f32 ] t
val f64 : [ `f64 ] t
val c16 : [ `c16 ] t
val c32 : [ `c32 ] t
val c64 : [ `c64 ] t
val bool : [ `bool ] t

type packed = T : _ t -> packed

val to_int : _ t -> int
val packed_to_int : packed -> int
val of_int_exn : int -> packed
val ( <> ) : packed -> packed -> bool
val size_in_bytes : packed -> int

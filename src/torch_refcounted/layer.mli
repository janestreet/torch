open Base

(** {3 Layer Types and Conversions} *)

(** A layer takes as input a tensor and returns a tensor through the [forward] function.
    Layers can hold variables, these are created and registered using a [Var_store.t] when
    creating the layer. *)
type t

(** A layer of type [t_with_training] is similar to a layer of type [t] except that it is
    also given a boolean argument when applying it to a tensor that specifies whether the
    layer is currently used in training or in testing mode. This is typically the case for
    batch normalization or dropout. *)
type t_with_training

(** [with_training t] returns a layer using the [is_training] argument from a standard
    layer. The [is_training] argument is discarded. This is useful when sequencing
    multiple layers via [fold]. *)
val with_training : t -> t_with_training

(** {3 Basic Layer Creation} *)

(** The identity layer. [forward id tensor] returns [tensor]. *)
val id : t

(** The identity layer with an [is_training] argument. *)
val id_ : t_with_training

(** [of_fn f] creates a layer based on a function from tensors to tensors. *)
val of_fn : (Tensor.t -> Tensor.t) -> t

(** [of_fn_ f] creates a layer based on a function from tensors to tensors. [f] also has
    access to the [is_training] flag. *)
val of_fn_ : (Tensor.t -> is_training:bool -> Tensor.t) -> t_with_training

(** [sequential ts] applies sequentially a list of layers [ts]. *)
val sequential : t list -> t

(** [sequential_ ts] applies sequentially a list of layers [ts]. *)
val sequential_ : t_with_training list -> t_with_training

(** [forward t tensor] applies layer [t] to [tensor]. *)
val forward : t -> Tensor.t -> Tensor.t

(** [forward_ t tensor ~is_training] applies layer [t] to [tensor] with the specified
    [is_training] flag. *)
val forward_ : t_with_training -> Tensor.t -> is_training:bool -> Tensor.t

(** {3 Linear and Convolution Layers} *)

(** The different kind of activations supported by the various layers below. *)
type activation =
  | Relu
  | Gelu
  | Softmax
  | Log_softmax
  | Tanh
  | Leaky_relu
  | Sigmoid
  | Hardsigmoid

(** [linear vs ~input_dim output_dim] returns a linear layer. When using [forward], the
    input tensor has to use a shape of [batch_size * input_dim]. The returned tensor has a
    shape [batch_size * output_dim]. *)
val linear
  :  Var_store.t
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> input_dim:int
  -> int
  -> t

(** [conv1d vs ~ksize ~stride ~input_dim output_dim] returns a 1D convolution layer.
    [ksize] specifies the kernel size and [stride] the stride. When using [forward], the
    input tensor should have a shape [batch_size * input_dim * l] and the returned tensor
    will have a shape [batch_size * output_dim * l']. *)
val conv1d
  :  Var_store.t
  -> ksize:int
  -> stride:int
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> ?padding:int
  -> ?groups:int
  -> ?dilation:int
  -> input_dim:int
  -> int
  -> t

(** [conv_transpose1d] creates a 1D transposed convolution layer. This is sometimes also
    called 'deconvolution'. *)
val conv_transpose1d
  :  Var_store.t
  -> ksize:int
  -> stride:int
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> ?padding:int
  -> ?output_padding:int
  -> ?groups:int
  -> ?dilation:int
  -> input_dim:int
  -> int
  -> t

(** [conv2d vs ~ksize ~stride ~input_dim output_dim] returns a 2D convolution layer.
    [ksize] specifies the kernel size and [stride] the stride. When using [forward], the
    input tensor should have a shape [batch_size * input_dim * h * w] and the returned
    tensor will have a shape [batch_size * output_dim * h' * w']. *)
val conv2d
  :  Var_store.t
  -> ksize:int * int
  -> stride:int * int
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> ?padding:int * int
  -> ?groups:int
  -> ?dilation:int * int
  -> input_dim:int
  -> int
  -> t

(** [conv2d_] is similar to [conv2d] but uses the same kernel size, stride, padding, and
    dilation on both the height and width dimensions, so a single integer needs to be
    specified for these parameters. *)
val conv2d_
  :  Var_store.t
  -> ksize:int
  -> stride:int
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> ?padding:int
  -> ?groups:int
  -> ?dilation:int
  -> input_dim:int
  -> int
  -> t

(** [conv_transpose2d] creates a 2D transposed convolution layer. This is sometimes also
    called 'deconvolution'. *)
val conv_transpose2d
  :  Var_store.t
  -> ksize:int * int
  -> stride:int * int
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> ?padding:int * int
  -> ?output_padding:int * int
  -> ?groups:int
  -> ?dilation:int * int
  -> input_dim:int
  -> int
  -> t

(** [conv_transpose2d_] is similar to [conv_transpose2d] but uses a single value for the
    height and width dimension for the kernel size, stride, padding, output padding, and
    dilation. *)
val conv_transpose2d_
  :  Var_store.t
  -> ksize:int
  -> stride:int
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> ?padding:int
  -> ?output_padding:int
  -> ?groups:int
  -> ?dilation:int
  -> input_dim:int
  -> int
  -> t

(** [conv3d vs ~ksize ~stride ~input_dim output_dim] returns a 3D convolution layer.
    [ksize] specifies the kernel size and [stride] the stride. When using [forward], the
    input tensor should have a shape [batch_size * input_dim * h * w * d] and the returned
    tensor will have a shape [batch_size * output_dim * h' * w' * d']. *)
val conv3d
  :  Var_store.t
  -> ksize:int * int * int
  -> stride:int * int * int
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> ?padding:int * int * int
  -> ?groups:int
  -> ?dilation:int * int * int
  -> input_dim:int
  -> int
  -> t

(** [conv3d_] is similar to [conv3d] but uses the same kernel size, stride, padding, and
    dilation on the height, width, and depth dimensions, so a single integer needs to be
    specified for these parameters. *)
val conv3d_
  :  Var_store.t
  -> ksize:int
  -> stride:int
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> ?padding:int
  -> ?groups:int
  -> ?dilation:int
  -> input_dim:int
  -> int
  -> t

(** [conv_transpose3d] creates a 3D transposed convolution layer. This is sometimes also
    called 'deconvolution'. *)
val conv_transpose3d
  :  Var_store.t
  -> ksize:int * int * int
  -> stride:int * int * int
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> ?padding:int * int * int
  -> ?output_padding:int * int * int
  -> ?groups:int
  -> ?dilation:int * int * int
  -> input_dim:int
  -> int
  -> t

(** [conv_transpose3d_] is similar to [conv_transpose3d] but uses a single value for the
    height, width, and depth dimension for the kernel size, stride, padding, output
    padding, and dilation. *)
val conv_transpose3d_
  :  Var_store.t
  -> ksize:int
  -> stride:int
  -> ?activation:activation (* default: no activation *)
  -> ?use_bias:bool (* default: true *)
  -> ?w_init:Var_store.Init.t
  -> ?padding:int
  -> ?output_padding:int
  -> ?groups:int
  -> ?dilation:int
  -> input_dim:int
  -> int
  -> t

(** {3 Normalization} *)

(** [batch_norm2d vs dim] creates a batch norm 2D layer. This layer applies Batch
    Normalization over a 4D input [batch_size * dim * h * w]. The returned tensor has the
    same shape. *)
val batch_norm2d
  :  Var_store.t
  -> ?w_init:Var_store.Init.t
  -> ?cudnn_enabled:bool
  -> ?eps:float
  -> ?momentum:float
  -> int
  -> t_with_training

val layer_norm : Var_store.t -> ?cudnn_enable:bool -> ?eps:float -> int -> t

(** {3 Embeddings} *)

val embeddings
  :  ?sparse:bool
  -> ?scale_grad_by_freq:bool
  -> Var_store.t
  -> num_embeddings:int
  -> embedding_dim:int
  -> t

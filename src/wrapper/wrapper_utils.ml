let keep_values_alive vs = ignore (Sys.opaque_identity vs : 'a list)

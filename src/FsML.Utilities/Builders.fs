namespace FsML.Utilities

module Builders =

  type EitherBuilder () =
    member this.Bind(x, f) =
      match x with
      | Ok s -> f s
      | Error f -> Error f
    member this.Return x = Ok(x)
    member this.ReturnFrom x = x

  let either = EitherBuilder ()

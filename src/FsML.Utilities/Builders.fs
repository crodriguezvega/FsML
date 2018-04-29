namespace FsML.Utilities

module Builders =

  type EitherBuilder () =
    member this.Bind(x, f) =
      match x with
      | Ok s -> f s
      | Error f -> Error f
    member this.Return x = Ok(x)
    member this.ReturnFrom x = x
    member this.Zero() = Ok()
    member this.Delay(f) = f
    member this.Combine(x, f) = this.Bind(x, f)
    member this.Run(f) = f()
    member this.While(guard, body) =
      if guard () then this.Bind(body (), fun _ -> this.While(guard, body))
      else this.Zero()

  let either = EitherBuilder ()

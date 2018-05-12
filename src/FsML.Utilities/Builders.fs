namespace FsML.Utilities

module Builders =

    module Either =

        let inline private bind f x =
            match x with
            | Ok s -> f s
            | Error e -> Error e

        type EitherBuilder () = 
            member __.Zero () = Ok ()
            member __.Bind (x, f) = bind f x
            member __.Return x = Ok x
            member __.ReturnFrom x = x
            member __.Yield x = Ok x
            member __.YieldFrom x = x
            member __.Combine (x, f) = bind f x
            member __.Delay f = f
            member __.Run f = f ()
            member __.TryWith (body, handler) =
                try body ()
                with | ex -> handler ex
            member __.TryFinally (body, compensation) =
                try body ()
                finally compensation ()
            member this.While (guard, body) =
                if guard () then bind (fun () -> this.While(guard, body)) (body ())
                else this.Zero ()

        let either = EitherBuilder ()

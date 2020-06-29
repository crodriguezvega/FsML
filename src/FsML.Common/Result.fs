namespace FsML.Common

module ResultExtension =

  let apply fResult xResult =
    match fResult, xResult with
    | Ok f, Ok x ->
      Ok (f x)
    | Error errors, Ok _ ->
      Error errors
    | Ok _, Error errors ->
      Error errors
    | Error errors, Error errors' ->
      Error (List.concat [errors; errors'])

module ResultAlias =
 
  let (<!>) = Result.map
  let (<*>) = ResultExtension.apply
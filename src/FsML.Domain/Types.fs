namespace FsML.Domain

module Types =

  type ErrorResult =
    | InvalidValue of string
    | InvalidDimensions of string
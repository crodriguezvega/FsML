namespace FsML.Common

module Types =

  type ErrorResult =
  | InvalidValue of string
  | InvalidDimensions of string
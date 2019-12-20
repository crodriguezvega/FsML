namespace FsML.Common

open MathNet.Numerics.LinearAlgebra

module Types =

  type Regressors = Matrix<float>
  type Regressand = Vector<float>
  type Weights = Vector<float>
  type Predictions = Vector<float>
  type MSE = float
  type RSquared = float
  type AdjustedRSquared = float

  type ErrorResult =
  | InvalidValue of string
  | InvalidDimensions of string
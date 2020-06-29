namespace FsML.Domain

open MathNet.Numerics.LinearAlgebra

open Types

module Regression =

  type Regressors = Matrix<float>
  type Regressand = Vector<float>
  type Weights = Vector<float>
  type Predictions = Vector<float>
  type MSE = float
  type RSquared = float
  type AdjustedRSquared = float

  type Regularization =
  | Without
  | With of float
  
  type TrainingParameters = private {
    H: Regressors
    Y: Regressand
  }

  type PredictionParameters = private {
    H: Regressors
    W: Weights
  }

  type GoodnessOfFitParameters = private {
    H: Regressors
    Y: Regressand
    W: Weights
  }

  type CostParameters = private {
    H: Regressors
    Y: Regressand
    W: Weights
  }

  type HypothesisFunction = Regressors -> Weights -> Predictions
  type CostFunction = Regularization -> Regressors -> Regressand -> Weights -> MSE
  
  module TrainingParameters =

    let create (H: Regressors) (Y: Regressand) : Result<TrainingParameters, ErrorResult list> =
      if H.RowCount <> Y.Count then
        Error [(InvalidDimensions "The number of observations and observed values must be the same")]
      else
        Ok ({ H = H; Y = Y })

    let getH (parameters: TrainingParameters) : Regressors = parameters.H
    let getY (parameters: TrainingParameters) : Regressand = parameters.Y

  module PredictionParameters =

    let create (H: Regressors) (W: Weights) : Result<PredictionParameters, ErrorResult list> =
      if H.ColumnCount <> W.Count then
        Error [(InvalidDimensions "The number of observations and weights must be the same")]
      else
        Ok ({ H = H; W = W })

    let getH (parameters: PredictionParameters) : Regressors = parameters.H
    let getW (parameters: PredictionParameters) : Weights = parameters.W

  module GoodnessOfFitParameters =

    let create (H: Regressors) (Y: Regressand) (W: Weights) : Result<GoodnessOfFitParameters, ErrorResult list> =
      if H.RowCount <> Y.Count then
        Error [(InvalidDimensions "The number of observations and observed values must be the same")]
      elif H.ColumnCount <> W.Count then
        Error [(InvalidDimensions "The number of observations and weights must be the same")]
      else
        Ok ({ H = H; Y = Y; W = W })

    let getH (parameters: GoodnessOfFitParameters) : Regressors = parameters.H
    let getY (parameters: GoodnessOfFitParameters) : Regressand = parameters.Y
    let getW (parameters: GoodnessOfFitParameters) : Weights = parameters.W

  module CostParameters =

    let create (H: Regressors) (Y: Regressand) (W: Weights) : Result<CostParameters, ErrorResult list> =
      if H.RowCount <> Y.Count then
        Error [(InvalidDimensions "The number of observations and observed values must be the same")]
      elif H.ColumnCount <> W.Count then
        Error [(InvalidDimensions "The number of observations and weights must be the same")]
      else
        Ok ({ H = H; Y = Y; W = W })

    let getH (parameters: CostParameters) : Regressors = parameters.H
    let getY (parameters: CostParameters) : Regressand = parameters.Y
    let getW (parameters: CostParameters) : Regressand = parameters.W

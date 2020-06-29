namespace FsML.Common

open ResultAlias
open MathNet.Numerics.LinearAlgebra


type ErrorResult =
    | InvalidValue of string
    | InvalidDimensions of string


module Types =

  type ErrorResult =
    | InvalidValue of string
    | InvalidDimensions of string

  module Regression =

    type Regressors = Matrix<float>
    type Regressand = Vector<float>
    type Weights = Vector<float>
    type Predictions = Vector<float>
    type Gradient = Vector<float>
    type MSE = float
    type RSquared = float
    type AdjustedRSquared = float

    type Regularization =
    | Without
    | With of float

    type GradientDescent =
    | Batch
    | Stochastic

    type Tolerance = private Tolerance of float
    type LearningRate = private LearningRate of float
    type NumberOfIterations = private NumberOfIterations of uint32

    type TrainingParameters = private {
      H: Regressors
      Y: Regressand
    }

    type PredictionParameters = private {
      H: Regressors
      Y: Regressand
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

    type GradientDescentParameters = {
      category: GradientDescent;
      tolerance: Tolerance;
      learningRate: LearningRate; // Step size
      numberOfIterations: NumberOfIterations
    }

    type HypothesisFunction = Regressors -> Weights -> Predictions

    module Tolerance = 

      let create (tolerance: float) : Result<Tolerance, ErrorResult list> =
        if tolerance <= 0.0 then
          Error [(InvalidDimensions "The tolerance must be greater than zero")]
        else
          Ok (Tolerance tolerance)

      let value (Tolerance tolerance) = tolerance

    module LearningRate = 

      let create (learningRate: float): Result<LearningRate, ErrorResult list> =
        if learningRate <= 0.0 then
          Error [(InvalidDimensions "The learning rate must be greater than zero")]
        else
          Ok (LearningRate learningRate)

      let value (LearningRate learningRate) = learningRate

    module NumberOfIterations = 

      let create (numberOfIterations: uint32) : Result<NumberOfIterations, ErrorResult list> =
        if numberOfIterations = 0u then
          Error [(InvalidDimensions "The number of iterations must be greater than zero")]
        else
          Ok (NumberOfIterations numberOfIterations)

      let value (NumberOfIterations numberOfIterations) = numberOfIterations

    module GradientDescentParameters =

      let create
        (category: GradientDescent)
        (tolerance: float) 
        (learningRate: float) 
        (numberOfIterations: uint32) 
        : Result<GradientDescentParameters, ErrorResult list> =

        let gradientDescent: Result<GradientDescent, ErrorResult list> = Ok (category)
        let doCreate category tolerance learningRate numberOfIterations : GradientDescentParameters =
          { category = category;
            tolerance = tolerance;
            learningRate = learningRate;
            numberOfIterations = numberOfIterations }
            
        doCreate 
          <!> gradientDescent
          <*> Tolerance.create tolerance
          <*> LearningRate.create learningRate
          <*> NumberOfIterations.create numberOfIterations

      let getLearningRate (parameters: GradientDescentParameters) : float = 
        LearningRate.value parameters.learningRate
      let getTolerance (parameters: GradientDescentParameters) : float = 
        Tolerance.value parameters.tolerance
      let getNumberOfIterations (parameters: GradientDescentParameters) : uint32 = 
        NumberOfIterations.value parameters.numberOfIterations

    module TrainingParameters =

      let create (H: Regressors) (Y: Regressand) : Result<TrainingParameters, ErrorResult> =
        if H.RowCount <> Y.Count then
          Error (InvalidDimensions "The number of observations and observed values must be the same")
        else
          Ok ({ H = H; Y = Y })

      let getH (parameters: TrainingParameters) : Regressors = parameters.H
      let getY (parameters: TrainingParameters) : Regressand = parameters.Y

    module PredictionParameters =

      let create (H: Regressors) (Y: Regressand) : Result<PredictionParameters, ErrorResult> =
        if H.RowCount <> Y.Count then
          Error (InvalidDimensions "The number of observations and observed values must be the same")
        else
          Ok ({ H = H; Y = Y })

    module GoodnessOfFitParameters =

      let create (H: Regressors) (Y: Regressand) (W: Weights) : Result<GoodnessOfFitParameters, ErrorResult> =
        if H.RowCount <> Y.Count then
          Error (InvalidDimensions "The number of observations and observed values must be the same")
        elif H.ColumnCount <> W.Count then
          Error (InvalidDimensions "The number of observations and weights must be the same")
        else
          Ok ({ H = H; Y = Y; W = W })

    module CostParameters =

      let create (H: Regressors) (Y: Regressand) (W: Weights) : Result<CostParameters, ErrorResult> =
        if H.RowCount <> Y.Count then
          Error (InvalidDimensions "The number of observations and observed values must be the same")
        elif H.ColumnCount <> W.Count then
          Error (InvalidDimensions "The number of observations and weights must be the same")
        else
          Ok ({ H = H; Y = Y; W = W })

      let getH (parameters: CostParameters) : Regressors = parameters.H
      let getY (parameters: CostParameters) : Regressand = parameters.Y
      let getW (parameters: CostParameters) : Regressand = parameters.W
      
namespace FsML.Domain

open MathNet.Numerics.LinearAlgebra

open Types
open Regression
open FsML.Common.ResultAlias

module Optimization =

  type Gradient = Vector<float>

  type GradientDescent =
  | Batch
  | Stochastic

  type Tolerance = private Tolerance of float
  type LearningRate = private LearningRate of float
  type NumberOfIterations = private NumberOfIterations of uint32

  type GradientDescentParameters = {
    category: GradientDescent;
    tolerance: Tolerance;
    learningRate: LearningRate; // Step size
    numberOfIterations: NumberOfIterations
  }

  type GradientOfCostFunction = 
    Regularization 
      -> GradientDescent 
      -> HypothesisFunction
      -> Regressors
      -> Regressand
      -> Weights
      -> Gradient

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

namespace FsML.Algorithms

open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

open FsML.Common.Builders
open FsML.Domain.Types
open FsML.Domain.Regression
open FsML.Domain.Optimization

module Optimization =

  /// <summary>
  /// Gradient of cost function
  /// </summary>
  /// <remarks>
  /// H: Matrix of observations (observation per row and feature per column) (a.k.a. regressors)
  /// Y: Vector of observed values (a.k.a. regressand)
  /// W: Vector of weights
  /// - For batch gradient descent:
  ///   - Without regularization: gradient = (1 / m) * (Hᵀ) * (H * W - Y)
  ///   - With regularization:    gradient = (1 / m) * ((Hᵀ) * (H * W - Y) + λ * [0, W[1:end]])
  /// For stochastic gradient descent:
  ///   - Without regularization: gradient = H[0,:] * ((H[0,:] * W)[0] - Y[0])
  ///   - With regularization:    gradient = H[0,:] * ((H[0,:] * W)[0] - Y[0] + λ * [0, W[1:end]])
  /// where m = number of observations
  ///       λ = regularization factor
  /// </remarks>
  /// <param name="regularization">Regularization flag</param>
  /// <param name="gradientDescent">Regularization flag</param>
  /// <param name="hypothesis">Hypothesis function</param>
  /// <param name="H">Matrix of observations (observation per row and feature per column) (a.k.a. regressors)</param>
  /// <param name="Y">Vector of observed values (a.k.a. regressand)</param>
  /// <param name="W">Vector of weights</param>
  let private gradientOfCostFunction
    (regularization:  Regularization)
    (gradientDescent: GradientDescent)
    (hypothesis:      HypothesisFunction)
    (H:               Regressors)
    (Y:               Regressand)
    (W:               Weights)
    : Gradient =

    let calculate aux (gradientWithoutRegularization: Vector<float>) =
      match regularization with
      | Regularization.Without ->
        gradientWithoutRegularization
      | Regularization.With(lambda) ->
        let regularizationTerm = aux * lambda * (Array.append [|0.0|] (W.SubVector(1, W.Count - 1).ToArray()) |> DenseVector.ofArray)
        gradientWithoutRegularization + regularizationTerm

    match gradientDescent with
    | GradientDescent.Batch ->
      let aux = 1.0 / float H.RowCount
      let prediction = hypothesis H W
      let gradientWithoutRegularization = aux * H.TransposeThisAndMultiply(prediction - Y)
      calculate aux gradientWithoutRegularization
    | GradientDescent.Stochastic ->
      let aux = 1.0
      let trainingSample = H.Row(0)
      let outputSample = Y.At(0)
      let prediction = hypothesis (DenseMatrix.ofRows([trainingSample])) W
      let gradientWithoutRegularization = trainingSample.Multiply(prediction.At(0) - outputSample)
      calculate aux gradientWithoutRegularization     

  /// <summary>
  /// Batch gradient descent
  /// </summary>
  let private calculateWeightWithBGD 
    (regularization: Regularization)
    (parameters:     GradientDescentParameters)
    (hypothesis:     HypothesisFunction)
    (X:              Regressors)
    (Y:              Regressand)
    (initialWeights: Weights)
    : Vector<float> =

    let category = parameters.category
    let learningRate = GradientDescentParameters.getLearningRate parameters

    let gradient = gradientOfCostFunction regularization category hypothesis X Y initialWeights
    initialWeights - gradient.Multiply(learningRate)

  /// <summary>
  /// Stochastic gradient descent
  /// </summary>
  let private calculateWeightWithSGD 
    (regularization: Regularization)
    (parameters:     GradientDescentParameters)
    (hypothesis:     HypothesisFunction)
    (X:              Regressors)
    (Y:              Regressand)
    (beginWeights:   Weights)
    : Weights =

    let category = parameters.category
    let learningRate = GradientDescentParameters.getLearningRate parameters

    let trainingSamples = (List.ofSeq (X.EnumerateRows()))
    let outputSamples = List.ofArray (Y.ToArray())
    let rec loop beginWeights samples =
      match samples with
      | [] -> beginWeights
      | head :: tail -> 
        let trainingSamples, outputSample = head
        let x = (DenseMatrix.OfRowVectors([trainingSamples]))
        let y = (DenseVector.OfArray([|outputSample|]))
        let gradient = gradientOfCostFunction regularization category hypothesis x y beginWeights
        let endWeights = beginWeights - learningRate * gradient
        loop endWeights tail

    let samples = List.zip trainingSamples outputSamples
    loop beginWeights samples

  /// <summary>
  /// Gradient descent
  /// </summary>
  let gradientDescent
    (regularization:     Regularization)
    (parameters:         GradientDescentParameters)
    (hypothesis:         HypothesisFunction)
    (costFunction:       CostFunction)
    (trainingParameters: TrainingParameters)
    : Weights =

    let H = TrainingParameters.getH trainingParameters
    let Y = TrainingParameters.getY trainingParameters
    let tolerance = GradientDescentParameters.getTolerance parameters
    let numberOfIterations = GradientDescentParameters.getNumberOfIterations parameters

    let initialCostDifference = 1.0
    let weights = Vector<float>.Build.Dense(H.ColumnCount)

    let gradientDescentOperation =
      match parameters.category with
      | Batch -> calculateWeightWithBGD regularization parameters hypothesis H Y
      | Stochastic -> calculateWeightWithSGD regularization parameters hypothesis H Y

    let rec calculate costDifference numberOfIterations weights =
      if (Double.IsNaN(costDifference) || costDifference <= tolerance || numberOfIterations = 0u) then
        weights
      else
        let beginCost = costFunction regularization H Y weights
        let newWeights = gradientDescentOperation weights
        let endCost = costFunction regularization H Y newWeights
        calculate (beginCost - endCost) (numberOfIterations - 1u) newWeights

    calculate initialCostDifference numberOfIterations weights

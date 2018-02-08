﻿namespace FsML.Algorithms.Regression

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

open FsML.Algorithms

module LinearRegression =

  /// Hypothesis
  let hypothesys (X: Matrix<float>) (theta: Vector<float>) = X * theta

  /// <summary>
  /// Cost function (residual sum of squares)
  /// </summary>
  /// <remarks>
  /// - Without regularization: cost = (1 / (2 * m)) (Y - H * W) ^ 2
  /// - With regularization:    cost = (1 / (2 * m)) ((Y - H * W) ^ 2 + λ * Σ(W[1:end] ^ 2)
  /// where m = number of observations
  ///       λ = regularization factor
  /// </remarks>
  /// <param name="Y">Vector of observed values</param>
  /// <param name="H">Matrix of observations (observation per row and feature per column)</param>
  /// <param name="W">Vector of weights</param>
  /// <returns>rss</returns>
  let costFunction regularization (X: Matrix<float>) (Y: Vector<float>) (W: Vector<float>) =
    let aux = (1.0 / (2.0 * float X.RowCount))
    let costWithoutRegularization = aux * ((hypothesys X W) - Y).PointwisePower(2.0).Sum()
    match regularization with
    | Optimization.Regularization.Without ->
      costWithoutRegularization
    | Optimization.Regularization.With(lambda) ->
      let regularizationTerm = aux * lambda * W.SubVector(1, W.Count - 1).PointwisePower(2.0).Sum()
      costWithoutRegularization + regularizationTerm

  /// Gradient of cost function
  let gradientOfCostFunction regularization gradientDescent (X: Matrix<float>) (Y: Vector<float>) (theta: Vector<float>) =
    let calculate aux (gradientWithoutRegularization: Vector<float>) =
      match regularization with
      | Optimization.Regularization.Without ->
        gradientWithoutRegularization
      | Optimization.Regularization.With(lambda) ->
        let regularizationTerm = aux * lambda * (Array.append [|0.0|] (theta.SubVector(1, theta.Count - 1).ToArray()) |> DenseVector.OfArray)
        gradientWithoutRegularization + regularizationTerm

    match gradientDescent with
    | Optimization.GradientDescent.Standard ->
      let aux = 1.0 / float X.RowCount
      let gradientWithoutRegularization = aux * X.TransposeThisAndMultiply((hypothesys X theta) - Y)
      calculate aux gradientWithoutRegularization
    | Optimization.GradientDescent.Stochastic ->
      let aux = 1.0
      let trainingSample = X.Row(0)
      let outputSample = Y.At(0)
      let hypothesisOutput = hypothesys (DenseMatrix.OfRowVectors([trainingSample])) theta
      let gradientWithoutRegularization = trainingSample.Multiply(hypothesisOutput.At(0) - outputSample)
      calculate aux gradientWithoutRegularization

  /// <summary>
  /// Fit with gradient descent
  /// </summary>
  /// <remarks>
  /// - If alpha too small -> slow convergence
  /// - If alpha too large -> may not converge
  /// </remarks>
  /// <param name="Y">Vector of observed values</param>
  /// <param name="H">Matrix of observations (observation per row and feature per column)</param>
  /// <param name="W">Vector of weights</param>
  /// <returns>rss</returns>
  let fitWithGradientDescent regularization gradientDescent (X: Matrix<float>) (Y: Vector<float>) learningRate numberOfiterations =
    Optimization.gradientDescent regularization gradientDescent costFunction gradientOfCostFunction X Y learningRate numberOfiterations

  /// <summary>
  /// Fit with normal equation
  /// </summary>
  /// <remarks>
  /// - Without regularization: W = inverse(transpose(H) * H) * transpose(H) * Y
  /// - With regularization:    W = 
  /// </remarks>
  /// <param name="regularization">Regularization flag</param>
  /// <param name="H">Matrix of observations (observation per row and feature per column)</param>
  /// <param name="Y">Vector of observed values</param>
  /// <returns>Vector of weights</returns>
  let fitWithNormalEquation regularization (H: Matrix<float>) (Y: Vector<float>) =
    let mainTerm = H.TransposeThisAndMultiply H
    match regularization with
    | Optimization.Regularization.Without ->
      (mainTerm.Inverse().TransposeAndMultiply H) * Y
    | Optimization.Regularization.With(lambda) ->
      let regularizationTerm = SparseMatrix.diag H.ColumnCount 1.0
      regularizationTerm.[0, 0] <- 0.0
      ((mainTerm + lambda * regularizationTerm).Inverse().TransposeAndMultiply H) * Y

  /// <summary>
  /// Predict
  /// </summary>
  /// <param name="H">Matrix of observations (observation per row and feature per column)</param>
  /// <param name="W">Vector of weights</param>
  /// <returns>Vector of predictions</returns>
  let predict (H: Matrix<float>) (W: Vector<float>) = hypothesys H W
﻿namespace FsML

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module LinearRegression =

  type Regularization =
    | Without
    | With of float

  /// Cost function
  let costFunction (trainingX: Matrix<_>) (trainingY: Vector<_>) (theta: Vector<_>) regularization =
    let aux = (1.0 / (2.0 * float trainingX.RowCount))
    let costWithoutRegularization = aux * (trainingX * theta - trainingY).PointwisePower(2.0).Sum()
    match regularization with
    | Without -> costWithoutRegularization
    | With(lambda) -> costWithoutRegularization + aux * lambda * theta.SubVector(1, theta.Count - 1).PointwisePower(2.0).Sum()

  /// Fit with gradient descent
  let fitWithGradientDescent (trainingX: Matrix<_>) (trainingY: Vector<_>) learningRate numberOfiterations regularization =
    let rec loop i (startTheta: Vector<_>) costDifference =
      let startCost = costFunction trainingX trainingY startTheta regularization
      match i, costDifference with
      | _, costDifference when (costDifference < 0.0 || Double.IsNaN(costDifference)) -> startTheta
      | 0, _ -> startTheta
      | _, _ ->
        let aux = learningRate / float trainingX.RowCount
        let mainTerm = trainingX.TransposeThisAndMultiply(trainingX * startTheta - trainingY)
        match regularization with
        | Without ->
          let endTheta = (startTheta - aux * mainTerm)
          let endCost = costFunction trainingX trainingY endTheta regularization
          loop (i - 1) endTheta endCost
        | With(lambda) ->
          let regularizationTerm = lambda * (Array.append [|0.0|] (startTheta.SubVector(1, startTheta.Count - 1).ToArray()) |> DenseVector.OfArray)
          let endTheta = startTheta - aux * (mainTerm + regularizationTerm)
          let endCost = costFunction trainingX trainingY endTheta regularization
          loop (i - 1) endTheta endCost
    loop numberOfiterations (DenseVector(trainingX.ColumnCount)) 1.0

  /// Fit with normal equation
  let fitWithNormalEquation (trainingX: Matrix<_>) (trainingY: Vector<_>) regularization =
    let mainTerm = trainingX.TransposeThisAndMultiply trainingX
    match regularization with
    | Without -> (mainTerm.Inverse().TransposeAndMultiply trainingX) * trainingY
    | With(lambda) ->
      let regularizationTerm = SparseMatrix.diag trainingX.ColumnCount 1.0
      regularizationTerm.[0, 0] <- 0.0
      ((mainTerm + lambda * regularizationTerm).Inverse().TransposeAndMultiply trainingX) * trainingY
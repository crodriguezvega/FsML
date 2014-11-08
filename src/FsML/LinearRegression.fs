namespace FsML

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module LinearRegression =

  /// Hypothesis
  let hypothesys (X: Matrix<_>) (theta: Vector<_>) = X * theta

  /// Cost function
  let costFunction (X: Matrix<_>) (Y: Vector<_>) (theta: Vector<_>) regularization =
    let aux = (1.0 / (2.0 * float X.RowCount))
    let costWithoutRegularization = aux * ((hypothesys X theta) - Y).PointwisePower(2.0).Sum()
    match regularization with
    | Optimization.Regularization.Without -> costWithoutRegularization
    | Optimization.Regularization.With(lambda) -> costWithoutRegularization + aux * lambda * theta.SubVector(1, theta.Count - 1).PointwisePower(2.0).Sum()

  /// Gradient of cost function
  let gradientOfCostFunction (X: Matrix<_>) (Y: Vector<_>) (theta: Vector<_>) regularization =
    let aux = 1.0 / float X.RowCount
    let mainTerm = X.TransposeThisAndMultiply((hypothesys X theta) - Y)
    match regularization with
    | Optimization.Regularization.Without -> aux * mainTerm
    | Optimization.Regularization.With(lambda) ->
      let regularizationTerm = lambda * (Array.append [|0.0|] (theta.SubVector(1, theta.Count - 1).ToArray()) |> DenseVector.OfArray)
      aux * (mainTerm + regularizationTerm)

  /// Fit with gradient descent
  let fitWithGradientDescent (trainingX: Matrix<_>) (trainingY: Vector<_>) learningRate numberOfiterations regularization =
    let trainingXWithInterceptTerm = trainingX.InsertColumn(0, DenseVector.Create(trainingX.RowCount, 1.0))
    Optimization.gradientDescent costFunction gradientOfCostFunction trainingXWithInterceptTerm trainingY learningRate numberOfiterations regularization

  /// Fit with normal equation
  let fitWithNormalEquation (trainingX: Matrix<_>) (trainingY: Vector<_>) regularization =
    let trainingXWithInterceptTerm = trainingX.InsertColumn(0, DenseVector.Create(trainingX.RowCount, 1.0))
    let mainTerm = trainingXWithInterceptTerm.TransposeThisAndMultiply trainingXWithInterceptTerm
    match regularization with
    | Optimization.Regularization.Without -> (mainTerm.Inverse().TransposeAndMultiply trainingXWithInterceptTerm) * trainingY
    | Optimization.Regularization.With(lambda) ->
      let regularizationTerm = SparseMatrix.diag trainingXWithInterceptTerm.ColumnCount 1.0
      regularizationTerm.[0, 0] <- 0.0
      ((mainTerm + lambda * regularizationTerm).Inverse().TransposeAndMultiply trainingXWithInterceptTerm) * trainingY
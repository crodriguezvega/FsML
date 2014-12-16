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
    let gradientWithoutRegularization = aux * X.TransposeThisAndMultiply((hypothesys X theta) - Y)
    match regularization with
    | Optimization.Regularization.Without -> gradientWithoutRegularization
    | Optimization.Regularization.With(lambda) ->
      let regularizationTerm = lambda * (Array.append [|0.0|] (theta.SubVector(1, theta.Count - 1).ToArray()) |> DenseVector.OfArray)
      gradientWithoutRegularization + aux * regularizationTerm

  /// Fit with gradient descent
  let fitWithGradientDescent (trainingX: Matrix<_>) (trainingY: Vector<_>) learningRate numberOfiterations regularization =
    Optimization.gradientDescent costFunction gradientOfCostFunction trainingX trainingY learningRate numberOfiterations regularization

  /// Fit with normal equation
  let fitWithNormalEquation (trainingX: Matrix<_>) (trainingY: Vector<_>) regularization =
    let mainTerm = trainingX.TransposeThisAndMultiply trainingX
    match regularization with
    | Optimization.Regularization.Without -> (mainTerm.Inverse().TransposeAndMultiply trainingX) * trainingY
    | Optimization.Regularization.With(lambda) ->
      let regularizationTerm = SparseMatrix.diag trainingX.ColumnCount 1.0
      regularizationTerm.[0, 0] <- 0.0
      ((mainTerm + lambda * regularizationTerm).Inverse().TransposeAndMultiply trainingX) * trainingY

  // Predict
  let predict (X: Matrix<_>) (theta: Vector<_>) = hypothesys X theta
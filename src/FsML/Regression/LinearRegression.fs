namespace FsML

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module LinearRegression =

  /// Hypothesis
  let hypothesys (X: Matrix<float>) (theta: Vector<float>) = X * theta

  /// Cost function
  let costFunction regularization (X: Matrix<float>) (Y: Vector<float>) (theta: Vector<float>) =
    let aux = (1.0 / (2.0 * float X.RowCount))
    let costWithoutRegularization = aux * ((hypothesys X theta) - Y).PointwisePower(2.0).Sum()
    match regularization with
    | Optimization.Regularization.Without ->
      costWithoutRegularization
    | Optimization.Regularization.With(lambda) ->
      let regularizationTerm = aux * lambda * theta.SubVector(1, theta.Count - 1).PointwisePower(2.0).Sum()
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

  /// Fit with gradient descent
  let fitWithGradientDescent regularization gradientDescent (X: Matrix<float>) (Y: Vector<float>) learningRate numberOfiterations =
    Optimization.gradientDescent regularization gradientDescent costFunction gradientOfCostFunction X Y learningRate numberOfiterations

  /// Fit with normal equation
  let fitWithNormalEquation regularization (X: Matrix<float>) (Y: Vector<float>) =
    let mainTerm = X.TransposeThisAndMultiply X
    match regularization with
    | Optimization.Regularization.Without ->
      (mainTerm.Inverse().TransposeAndMultiply X) * Y
    | Optimization.Regularization.With(lambda) ->
      let regularizationTerm = SparseMatrix.diag X.ColumnCount 1.0
      regularizationTerm.[0, 0] <- 0.0
      ((mainTerm + lambda * regularizationTerm).Inverse().TransposeAndMultiply X) * Y

  /// Predict
  let predict (X: Matrix<float>) (theta: Vector<float>) = hypothesys X theta
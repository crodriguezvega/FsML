namespace FsML

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module LogisticRegression =

  /// Sigmoid function
  let sigmoidFunction (Z: Vector<_>) =
    1.0 / (1.0 + (-1.0 * Z)).PointwiseExp()

  /// Hypothesis
  let hypothesys (X: Matrix<_>) (theta: Vector<_>) = sigmoidFunction (X * theta)

  /// Cost function
  let costFunction (X: Matrix<_>) (Y: Vector<_>) (theta: Vector<_>) regularization =
    let aux = 1.0 / float X.RowCount
    let costWithoutRegularization = -1.0 * aux * (Y * (sigmoidFunction (X * theta)).PointwiseLog()
                                    + (1.0 - Y) * (1.0 - sigmoidFunction (X * theta)).PointwiseLog())
    match regularization with
    | Optimization.Regularization.Without -> costWithoutRegularization
    | Optimization.Regularization.With(lambda) -> costWithoutRegularization 
                                                  + ((lambda * aux) / 2.0) * theta.SubVector(1, theta.Count - 1).PointwisePower(2.0).Sum()

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
    Optimization.gradientDescent costFunction gradientOfCostFunction trainingX trainingY learningRate numberOfiterations regularization

  // Predict
  let predict (X: Matrix<_>) (theta: Vector<_>) = hypothesys X theta
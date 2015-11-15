namespace FsML

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module LogisticRegression =

  /// Sigmoid function
  let sigmoidFunction (Z: Vector<float>) =
    1.0 / (1.0 + (-1.0 * Z).PointwiseExp())

  /// Hypothesis (output is the estimated probability that y = 1 on input x)
  let hypothesys (X: Matrix<float>) (theta: Vector<float>) = sigmoidFunction (X * theta)

  /// Cost function
  let costFunction regularization (X: Matrix<float>) (Y: Vector<float>) (theta: Vector<float>) =
    let aux = 1.0 / float X.RowCount
    let costWithoutRegularization = -1.0 * aux * (Y * (sigmoidFunction (X * theta)).PointwiseLog()
                                    + (1.0 - Y) * (1.0 - sigmoidFunction (X * theta)).PointwiseLog())
    match regularization with
    | Optimization.Regularization.Without ->
      costWithoutRegularization
    | Optimization.Regularization.With(lambda) ->
      let regularizationTerm = ((lambda * aux) / 2.0) * theta.SubVector(1, theta.Count - 1).PointwisePower(2.0).Sum()
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
  let fitWithGradientDescent regularization gradientDescent (trainingX: Matrix<float>) (trainingY: Vector<float>) learningRate numberOfiterations =
    Optimization.gradientDescent regularization gradientDescent costFunction gradientOfCostFunction trainingX trainingY learningRate numberOfiterations

  /// Predict
  let predict (X: Matrix<float>) (theta: Vector<float>) = Vector.map (fun x -> if x > 0.5 then 1.0 else 0.0 ) (hypothesys X theta)
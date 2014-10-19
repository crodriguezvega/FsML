namespace FsML

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module LogisticRegression =

  type Regularization =
    | Without
    | With of float

  /// Sigmoid function
  let sigmoidFunction (X: Matrix<_>) (theta: Vector<_>) =
    1.0 / (1.0 + (-1.0 * (X * theta)).PointwiseExp())

  /// Cost function
  let costFunction (trainingX: Matrix<_>) (trainingY: Vector<_>) (theta: Vector<_>) regularization =
    let aux = (1.0 / float trainingX.RowCount)
    let costWithoutRegularization = -1.0 * aux * (trainingY * (sigmoidFunction trainingX theta).PointwiseLog() + (1.0 - trainingY) * (1.0 - sigmoidFunction trainingX theta).PointwiseLog())
    match regularization with
    | Without -> costWithoutRegularization
    | With(lambda) -> costWithoutRegularization + ((lambda * aux) / 2.0) * theta.SubVector(1, theta.Count - 1).PointwisePower(2.0).Sum()

  /// Fit with gradient descent
  let fitWithGradientDescent (trainingX: Matrix<_>) (trainingY: Vector<_>) learningRate numberOfiterations regularization =
    let rec loop i (beginTheta: Vector<_>) costDifference =
      let beginCost = costFunction trainingX trainingY beginTheta regularization
      match i, costDifference with
      | _, costDifference when (costDifference < 0.0 || Double.IsNaN(costDifference)) -> beginTheta
      | 0, _ -> beginTheta
      | _, _ ->
        let aux = learningRate / float trainingX.RowCount
        let mainTerm = trainingX.TransposeThisAndMultiply(sigmoidFunction trainingX beginTheta - trainingY)
        match regularization with
        | Without ->
          let endTheta = (beginTheta - aux * mainTerm)
          let endCost = costFunction trainingX trainingY endTheta regularization
          loop (i - 1) endTheta (beginCost - endCost)
        | With(lambda) ->
          let regularizationTerm = lambda * (Array.append [|0.0|] (beginTheta.SubVector(1, beginTheta.Count - 1).ToArray()) |> DenseVector.OfArray)
          let endTheta = beginTheta - aux * (mainTerm + regularizationTerm)
          let endCost = costFunction trainingX trainingY endTheta regularization
          loop (i - 1) endTheta (beginCost - endCost)
    loop numberOfiterations (DenseVector(trainingX.ColumnCount)) 1.0

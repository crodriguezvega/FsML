namespace FsML

open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module LinearRegression =

  /// Fit with gradient descent
  let fitWithGradientDescent (trainingX: Matrix<_>) (trainingY: Vector<_>) learningRate numberOfiterations =
    let rec loop i (theta: Vector<_>) =
      match i with
      | 0 -> theta
      | _ ->
        let derivative = trainingX.TransposeThisAndMultiply (trainingX * theta - trainingY)
        loop (i - 1) (theta - (learningRate / float trainingX.ColumnCount) * derivative)
    loop numberOfiterations (DenseVector(trainingX.ColumnCount))

  /// Fit with normal equation
  let fitWithNormalEquation (trainingX: Matrix<_>) (trainingY: Vector<_>) =
    ((trainingX.TransposeThisAndMultiply trainingX).Inverse().TransposeAndMultiply trainingX) * trainingY
namespace FsML

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module Optimization =

  type Regularization =
    | Without
    | With of float

  // Gradient descent
  let gradientDescent costFunction (gradientOfCostFunction: Matrix<_> -> Vector<_> -> Vector<_> -> Regularization -> Vector<_>)
                      (X: Matrix<_>) (Y: Vector<_>) (learningRate: float) numberOfiterations regularization =
    let rec loop i (beginTheta: Vector<_>) costDifference =
      let beginCost = costFunction X Y beginTheta regularization
      match i, costDifference with
      | _, costDifference when (costDifference < 0.0 || Double.IsNaN(costDifference)) -> beginTheta
      | 0, _ -> beginTheta
      | _, _ ->
        let endTheta = (beginTheta - learningRate * (gradientOfCostFunction X Y beginTheta regularization))
        let endCost = costFunction X Y endTheta regularization
        loop (i - 1) endTheta (beginCost - endCost)
    loop numberOfiterations (DenseVector(X.ColumnCount)) 1.0
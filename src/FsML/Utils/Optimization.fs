namespace FsML

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module Optimization =

  type Regularization =
    | Without
    | With of float

  type GradientDescent =
    | Standard
    | Stochastic

  /// Standard gradient descent
  let calculateThetaForStandardGradientDescent regularization gradientDescent
                                               (gradientOfCostFunction: Regularization -> GradientDescent -> Matrix<float> -> Vector<float> -> Vector<float> -> Vector<float>)
                                               X Y (beginTheta: Vector<float>) (learningRate: float) =
    let gradient = gradientOfCostFunction regularization gradientDescent X Y beginTheta
    beginTheta - gradient.Multiply(learningRate)

  /// Calculation of 
  let calculateThetaForStochasticGradientDescent regularization gradientDescent gradientOfCostFunction (X: Matrix<float>) (Y: Vector<float>) beginTheta learningRate =
    let trainingSamples = (List.ofSeq (X.EnumerateRows()))
    let outputs = List.ofArray (Y.ToArray())
    let rec loop beginTheta trainingSamples outputs =
      match trainingSamples, outputs with
      | [], [] -> beginTheta
      | headTrainingSamples :: tailTrainingSamples, headOutputSamples :: tailOutputSamples ->
        let endTheta = beginTheta - learningRate * (gradientOfCostFunction regularization gradientDescent (DenseMatrix.OfRowVectors([headTrainingSamples])) (DenseVector.OfArray([|headOutputSamples|])) beginTheta)
        loop endTheta tailTrainingSamples tailOutputSamples
      | _, _ -> failwith "Number of training samples does not match number of outputs"
    loop beginTheta trainingSamples outputs

  /// Gradient descent for linear and logistic regression
  let gradientDescent regularization gradientDescent costFunction gradientOfCostFunction (X: Matrix<float>) (Y: Vector<float>) learningRate numberOfIterations =
    let rec loop i (beginTheta: Vector<float>) costDifference =
      let beginCost = costFunction regularization X Y beginTheta
      match i, costDifference with
      | _, costDifference when (costDifference < 0.0 || Double.IsNaN(costDifference)) -> beginTheta
      | 0, _ -> beginTheta
      | _, _ ->
        let endTheta = match gradientDescent with
                       | Standard ->
                         calculateThetaForStandardGradientDescent regularization gradientDescent gradientOfCostFunction X Y beginTheta learningRate
                       | Stochastic ->
                         calculateThetaForStochasticGradientDescent regularization gradientDescent gradientOfCostFunction X Y beginTheta learningRate
        let endCost = costFunction regularization X Y endTheta
        loop (i - 1) endTheta (beginCost - endCost)
    loop numberOfIterations (DenseVector(X.ColumnCount)) 1.0
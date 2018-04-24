namespace FsML.Algorithms

open System
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module Optimization =

  type Regularization =
    | Without
    | With of float

  type GradientDescent =
    | Batch
    | Stochastic

  type GradientDescentParameters =
    {
      category: GradientDescent;
      learningRate: float;
      numberOfiterations: int
    }
  
  type GradientOfCostFunction = Regularization -> GradientDescent -> Matrix<float> -> Vector<float> -> Vector<float> -> Vector<float>

  /// Batch gradient descent
  let calculateWeightForStandardGradientDescent regularization
                                                (gradientDescent: GradientDescentParameters)
                                                (gradientOfCostFunction: GradientOfCostFunction)
                                                X Y beginWeight =
    let gradient = gradientOfCostFunction regularization gradientDescent.category X Y beginWeight
    beginWeight - gradient.Multiply(gradientDescent.learningRate)

  /// Stochastic gradient descent
  let calculateWeightForStochasticGradientDescent regularization
                                                  (gradientDescent: GradientDescentParameters)
                                                  (gradientOfCostFunction: GradientOfCostFunction) 
                                                  (X: Matrix<float>) (Y: Vector<float>) beginWeight =
    let trainingSamples = (List.ofSeq (X.EnumerateRows()))
    let outputSamples = List.ofArray (Y.ToArray())
    let rec loop beginTheta trainingSamples outputSamples =
      match trainingSamples, outputSamples with
      | [], [] -> beginTheta
      | headTrainingSamples :: tailTrainingSamples, headOutputSamples :: tailOutputSamples ->
        let endTheta = beginTheta - gradientDescent.learningRate * (gradientOfCostFunction regularization gradientDescent.category (DenseMatrix.OfRowVectors([headTrainingSamples])) (DenseVector.OfArray([|headOutputSamples|])) beginWeight)
        loop endTheta tailTrainingSamples tailOutputSamples
      | _, _ -> failwith "Number of training samples does not match number of outputs"
    loop beginWeight trainingSamples outputSamples

  /// Gradient descent for linear and logistic regression
  let gradientDescent regularization (gradientDescent: GradientDescentParameters) costFunction gradientOfCostFunction X Y =
    let rec loop i beginWeight costDifference =
      let beginCost = costFunction regularization X Y beginWeight
      match i, costDifference with
      | _, costDifference when (costDifference < 0.0 || Double.IsNaN(costDifference)) -> beginWeight
      | 0, _ -> beginWeight
      | _, _ ->
        let endWeight = match gradientDescent.category with
                        | Batch ->
                          calculateWeightForStandardGradientDescent regularization gradientDescent gradientOfCostFunction X Y beginWeight
                        | Stochastic ->
                          calculateWeightForStochasticGradientDescent regularization gradientDescent gradientOfCostFunction X Y beginWeight
        let endCost = costFunction regularization X Y endWeight
        loop (i - 1) endWeight (beginCost - endCost)
    loop gradientDescent.numberOfiterations (DenseVector(X.ColumnCount)) 1.0

  // Find a better place for this
  let scaling (X: Vector<float>) =
    let mean = Statistics.Mean X
    let std = Statistics.StandardDeviation X
    match std with
    | 0.0 -> None
    | _ -> Some (X.Subtract(mean) / std)
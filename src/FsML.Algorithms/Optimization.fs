namespace FsML.Algorithms

open System
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

open FsML.Utilities
open FsML.Utilities.Builders

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
      numberOfiterations: uint32
    }
  
  type GradientOfCostFunction = Regularization -> GradientDescent -> Matrix<float> -> Vector<float> -> Vector<float> -> Vector<float>

  /// Batch gradient descent
  let calculateWeightWithBGD regularization (gradientDescent: GradientDescentParameters)
                             (gradientOfCostFunction: GradientOfCostFunction)
                             X Y beginWeight =
    let gradient = gradientOfCostFunction regularization gradientDescent.category X Y beginWeight
    Ok (beginWeight - gradient.Multiply(gradientDescent.learningRate))

  /// Stochastic gradient descent
  let calculateWeightWithSGD regularization
                             (gradientDescent: GradientDescentParameters)
                             (gradientOfCostFunction: GradientOfCostFunction) 
                             (X: Matrix<float>) (Y: Vector<float>) beginWeight =
    let trainingSamples = (List.ofSeq (X.EnumerateRows()))
    let outputSamples = List.ofArray (Y.ToArray())
    if (trainingSamples.Length <> outputSamples.Length) then
      Error Types.NumberTrainingSamplesMustEqualNumberOutputs
    else
      let rec loop beginTheta samples =
        match samples with
        | [] -> beginTheta
        | head :: tail -> 
          let trainingSamples, outputSamples = head
          let x = (DenseMatrix.OfRowVectors([trainingSamples]))
          let y = (DenseVector.OfArray([|outputSamples|]))
          let gradient = gradientOfCostFunction regularization gradientDescent.category x y beginWeight
          let endTheta = beginTheta - gradientDescent.learningRate * gradient
          loop endTheta tail

      let samples = List.zip trainingSamples outputSamples
      Ok (loop beginWeight samples)

  /// Gradient descent for linear and logistic regression
  let gradientDescent regularization (gradientDescent: GradientDescentParameters) costFunction gradientOfCostFunction (X: Matrix<float>) Y =
    let mutable costDifference = 1.0
    let mutable weight = Vector<float>.Build.Dense(X.ColumnCount)
    let mutable iteration = gradientDescent.numberOfiterations

    let costOperation = costFunction regularization X Y
    let gradientDescentOperation =
      match gradientDescent.category with
      | Batch -> calculateWeightWithBGD regularization gradientDescent gradientOfCostFunction X Y
      | Stochastic -> calculateWeightWithSGD regularization gradientDescent gradientOfCostFunction X Y

    let guard () = costDifference < 0.0 || Double.IsNaN(costDifference) || iteration = 0u
    Either.either {
      while guard () do
        let beginCost = costOperation weight
        let! result = gradientDescentOperation weight
        weight <- result
        let endCost = costOperation weight
        costDifference <- endCost - beginCost
        iteration <- iteration - 1u
      return weight
    }

  (*
    let rec loop (i: uint32) beginWeight costDifference =
      let beginCost = costFunction regularization X Y beginWeight
      match i, costDifference with
      | _, costDifference when (costDifference < 0.0 || Double.IsNaN(costDifference)) -> beginWeight
      | 0u, _ -> beginWeight
      | _, _ ->
        let endWeight = match gradientDescent.category with
          | Batch -> calculateWeightWithBGD regularization gradientDescent gradientOfCostFunction X Y beginWeight
          | Stochastic -> calculateWeightWithSGD regularization gradientDescent gradientOfCostFunction X Y beginWeight

        let endCost = costFunction regularization X Y endWeight
        loop (i - 1u) endWeight (beginCost - endCost)

    loop gradientDescent.numberOfiterations (DenseVector(X.ColumnCount)) 1.0
 *)
  // Find a better place for this
  let scaling (X: Vector<float>) =
    let mean = Statistics.Mean X
    let std = Statistics.StandardDeviation X
    match std with
    | 0.0 -> None
    | _ -> Some (X.Subtract(mean) / std)
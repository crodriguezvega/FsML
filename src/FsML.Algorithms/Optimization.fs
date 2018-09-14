namespace FsML.Algorithms

open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

open FsML.Common.Builders
open FsML.Common.Types

module Optimization =

    type Regularization =
    | Without
    | With of float

    type GradientDescent =
    | Batch
    | Stochastic

    type GradientDescentParameters = {
        category: GradientDescent;
        learningRate: float; // Step size
        numberOfIterations: uint32
    }

    type GradientOfCostFunction = Regularization -> GradientDescent -> Matrix<float> -> Vector<float> -> Vector<float> -> Vector<float>

    /// <summary>
    /// Batch gradient descent
    /// </summary>
    let private calculateWeightWithBGD regularization
                                       (parameters: GradientDescentParameters)
                                       (gradientOfCostFunction: GradientOfCostFunction)
                                       X
                                       Y
                                       beginWeight =
        let gradient = gradientOfCostFunction regularization parameters.category X Y beginWeight
        (beginWeight - gradient.Multiply(parameters.learningRate))

    /// <summary>
    /// Stochastic gradient descent
    /// </summary>
    let private calculateWeightWithSGD regularization
                                       (parameters: GradientDescentParameters)
                                       (gradientOfCostFunction: GradientOfCostFunction) 
                                       (X: Matrix<float>)
                                       (Y: Vector<float>)
                                       beginWeight =
        let trainingSamples = (List.ofSeq (X.EnumerateRows()))
        let outputSamples = List.ofArray (Y.ToArray())
        let rec loop beginWeight samples =
            match samples with
            | [] -> beginWeight
            | head :: tail -> 
                let trainingSamples, outputSample = head
                let x = (DenseMatrix.OfRowVectors([trainingSamples]))
                let y = (DenseVector.OfArray([|outputSample|]))
                let gradient = gradientOfCostFunction regularization parameters.category x y beginWeight
                let endWeight = beginWeight - parameters.learningRate * gradient
                loop endWeight tail

        let samples = List.zip trainingSamples outputSamples
        loop beginWeight samples

    /// <summary>
    /// Gradient descent
    /// </summary>
    let gradientDescent regularization
                        (parameters: GradientDescentParameters)
                        costFunction
                        gradientOfCostFunction
                        (X: Matrix<float>)
                        (Y: Vector<float>) =
        if X.RowCount <> Y.Count then
            Error (InvalidDimensions "The number of rows of X and the length of Y must be the same")
        elif parameters.learningRate <= 0.0 then
            Error (InvalidValue "The learning rate must be > 0")
        else
            let mutable costDifference = 1.0
            let mutable weight = Vector<float>.Build.Dense(X.ColumnCount)
            let mutable iteration = parameters.numberOfIterations
            let guard () = not (costDifference < (10.0 ** (-10.0)) || Double.IsNaN(costDifference) || iteration = 0u)

            Either.either {
                let costOperation = costFunction regularization X Y
                let gradientDescentOperation =
                    match parameters.category with
                    | Batch -> calculateWeightWithBGD regularization parameters gradientOfCostFunction X Y
                    | Stochastic -> calculateWeightWithSGD regularization parameters gradientOfCostFunction X Y

                while guard () do
                    let! beginCost = costOperation weight
                    let result = gradientDescentOperation weight
                    weight <- result
                    let! endCost = costOperation weight
                    costDifference <- beginCost - endCost
                    iteration <- iteration - 1u
                return weight
            }
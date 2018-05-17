namespace FsML.Algorithms.Tests

open Xunit
open MathNet.Numerics.LinearAlgebra

open FsML.Algorithms.Optimization

module OptimizationTests =

    [<Fact>]
    let ``Can calculate weight for batch gradient descent`` () =
        let gradientDescentParameters = {
            category = GradientDescent.Batch;
            learningRate = 1.0;
            numberOfiterations = 10u
        }
        let gradientOfCostFunctionStub (_: Regularization)
                                       (_: GradientDescent)
                                       (_: Matrix<float>)
                                       (_: Vector<float>)
                                       (_: Vector<float>) = [| 2.0; 3.0 |] |> Vector<float>.Build.DenseOfArray
        let X = [|
                    [| 0.0; 0.0|]
                    [| 0.0; 0.0|]
                |] |>Matrix<float>.Build.DenseOfRowArrays
        let Y = [| 0.0; 0.0 |] |> Vector<float>.Build.DenseOfArray
        let beginWeight = [| 3.0; 2.0 |] |> Vector<float>.Build.DenseOfArray

        let result = calculateWeightWithBGD Regularization.Without gradientDescentParameters gradientOfCostFunctionStub X Y beginWeight
        match result with
        | Error e -> Assert.True(false)
        | Ok weight -> Assert.Equal(1.0, weight.At(0))
                       Assert.Equal(-1.0, weight.At(1))


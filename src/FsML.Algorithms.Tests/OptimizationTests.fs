namespace FsML.Algorithms.Tests

open Xunit
open MathNet.Numerics.LinearAlgebra

open FsML.Algorithms.Optimization
open FsML.Common.Types

module OptimizationTests =

    [<Fact>]
    let ``Cannot execute gradient descent if number of rows of X and length of Y are not the same`` () =
        let regularization = Regularization.Without
        let parameters = {
            category = GradientDescent.Batch;
            learningRate = 0.0;
            numberOfIterations = 0u
        }
        let costFunction (_: Regularization)
                         (_: Matrix<float>)
                         (_: Vector<float>)
                         (_: Vector<float>) = Ok (0.0)
        let gradientOfCostFunction (_: Regularization)
                                   (_: GradientDescent)
                                   (_: Matrix<float>)
                                   (_: Vector<float>)
                                   (_: Vector<float>) = Vector<double>.Build.Random(2)

        let X = Matrix<double>.Build.Random(2, 1)
        let Y = Vector<double>.Build.Random(3)

        let result = gradientDescent regularization parameters costFunction gradientOfCostFunction X Y
        match result with
        | Error (InvalidDimensions _) -> Assert.True(true)
        | _ -> Assert.True(false)

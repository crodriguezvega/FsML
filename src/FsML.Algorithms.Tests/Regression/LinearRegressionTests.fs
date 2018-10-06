namespace FsML.Algorithms.Tests.Regression

open FsCheck
open FsCheck.Xunit
open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

open FsML.Algorithms
open FsML.Algorithms.Optimization
open FsML.Algorithms.Regression.LinearRegression
open FsML.Common.Builders
open FsML.Common.Types

module LinearRegressionTests =

    type Sample = { TrainingX: Matrix<float>; TrainingY: Vector<float> }

    type TestData =
        static member Sample () =
            gen {
                let normalDistribution = Normal.WithMeanVariance(0.0, 0.5)
                let! (intercept, slope) =  Gen.elements [-50.0..1.0..50.0] |> Gen.two

                let x = [| 1.0..1.0..10.0 |]
                let y = x |> Array.map (fun x -> (x * slope + intercept) + normalDistribution.Sample())

                let trainingX = [|
                                    (Array.create x.Length 1.0) |> Array.toList |> vector
                                    x |> Array.toList |> vector
                                |] |> DenseMatrix.OfColumnVectors
                let trainingY = y |> DenseVector.OfArray
                return { TrainingX = trainingX; TrainingY = trainingY }
            } |> Arb.fromGen

    [<Property(Arbitrary=[| typeof<TestData> |])>]
    let ``Can calculate regression line using normal equation`` ({ TrainingX = trainingX; TrainingY = trainingY }) =
        let epsilon = 0.5

        let fit: Result<Vector<float>, ErrorResult> = Either.either {
            let! fit = fitWithNormalEquation Optimization.Regularization.Without trainingX trainingY
            return fit
        }

        match fit with
        | Error _ -> false
        | Ok fit -> match costFunction Optimization.Regularization.Without trainingX trainingY fit with
                    | Error _ -> false
                    | Ok cost -> cost < 1.0

    [<Property(Arbitrary=[| typeof<TestData> |])>]
    let ``Can calculate regression line using batch gradient descent`` ({ TrainingX = trainingX; TrainingY = trainingY }) =
        let epsilon = 0.5

        let fit: Result<Vector<float>, ErrorResult> = Either.either {
            let gdParameters = { category = Optimization.GradientDescent.Batch; learningRate = 0.01; numberOfIterations = 5000u }
            let linearRegressionWithBGD = fitWithGradientDescent Optimization.Regularization.Without gdParameters
            let! fit = linearRegressionWithBGD trainingX trainingY
            return fit
        }

        match fit with
        | Error _ -> false
        | Ok fit -> match costFunction Optimization.Regularization.Without trainingX trainingY fit with
                    | Error _ -> false
                    | Ok cost -> cost < 1.0

    [<Property(Arbitrary=[| typeof<TestData> |])>]
    let ``Can calculate regression line using stochastic gradient descent`` ({ TrainingX = trainingX; TrainingY = trainingY }) =
        let fit: Result<Vector<float>, ErrorResult> = Either.either {
            let gdParameters = { category = Optimization.GradientDescent.Batch; learningRate = 0.01; numberOfIterations = 5000u }
            let linearRegressionWithBGD = fitWithGradientDescent Optimization.Regularization.Without gdParameters
            let! fit = linearRegressionWithBGD trainingX trainingY
            return fit
        }

        match fit with
        | Error _ -> false
        | Ok fit -> match costFunction Optimization.Regularization.Without trainingX trainingY fit with
                    | Error _ -> false
                    | Ok cost -> cost < 1.0
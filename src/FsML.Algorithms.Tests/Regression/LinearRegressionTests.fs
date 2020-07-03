namespace FsML.Algorithms.Tests.Regression

open FsCheck
open FsCheck.Xunit
open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

open FsML.Common.Builders
open FsML.Domain.Types
open FsML.Domain.Regression
open FsML.Domain.Optimization
open FsML.Algorithms.Regression.LinearRegression

module LinearRegressionTests =

  type Sample = { trainingH: Matrix<float>; trainingY: Vector<float> }

  type TestData =
    static member Sample () =
      gen {
        let normalDistribution = Normal.WithMeanVariance(0.0, 0.1)
        let! (intercept, slope) =  Gen.elements [-50.0 .. 1.0 .. 50.0] |> Gen.two

        let x = [| 1.0..1.0..10.0 |]
        let y = x |> Array.map (fun x -> (x * slope + intercept) + normalDistribution.Sample())

        let H = [|
                  (Array.create x.Length 1.0) |> Array.toList |> vector
                  x |> Array.toList |> vector
                |] |> DenseMatrix.OfColumnVectors
        let Y = y |> DenseVector.OfArray
        return { trainingH = H; trainingY = Y }
      } |> Arb.fromGen

  [<Property(Arbitrary=[| typeof<TestData> |])>]
  let ``Can calculate regression line using normal equation`` ({ trainingH = H; trainingY = Y }) =
    let epsilon = 0.5

    let cost: Result<MSE, ErrorResult list> = Either.either {
      let! trainingParameters = TrainingParameters.create H Y
      let fit = fitWithNormalEquation Regularization.Without trainingParameters
      let! costParameters = CostParameters.create H Y fit
      let cost = costFunction Regularization.Without costParameters
      return cost
    }

    match cost with
    | Error _ -> false
    | Ok cost -> cost < epsilon

  [<Property(Arbitrary=[| typeof<TestData> |])>]
  let ``Can calculate regression line using batch gradient descent`` ({ trainingH = H; trainingY = Y }) =
    let epsilon = 0.5
    
    let cost: Result<MSE, ErrorResult list> = Either.either {
      let! trainingParameters = TrainingParameters.create H Y
      let! gdParameters = GradientDescentParameters.create GradientDescent.Batch 0.0001 0.01 10000u
      let linearRegressionWithBGD = fitWithGradientDescent Regularization.Without gdParameters
      let fit = linearRegressionWithBGD trainingParameters
      let! costParameters = CostParameters.create H Y fit
      let cost = costFunction Regularization.Without costParameters
      return cost
    }

    match cost with
    | Error _ -> false
    | Ok cost -> cost < epsilon

  [<Property(Arbitrary=[| typeof<TestData> |])>]
  let ``Can calculate regression line using stochastic gradient descent`` ({ trainingH = H; trainingY = Y }) =
    let epsilon = 0.5

    let cost: Result<MSE, ErrorResult list> = Either.either {
      let! trainingParameters = TrainingParameters.create H Y
      let! gdParameters = GradientDescentParameters.create GradientDescent.Batch 0.0001 0.01 10000u
      let linearRegressionWithBGD = fitWithGradientDescent Regularization.Without gdParameters
      let fit = linearRegressionWithBGD trainingParameters
      let! costParameters = CostParameters.create H Y fit
      let cost = costFunction Regularization.Without costParameters
      return cost
    }

    match cost with
    | Error _ -> false
    | Ok cost -> cost < epsilon
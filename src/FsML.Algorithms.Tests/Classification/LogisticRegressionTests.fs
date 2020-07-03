namespace FsML.Algorithms.Tests.Regression

open Xunit
open FsCheck
open FsCheck.Xunit
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

open FsML.Common.Builders
open FsML.Domain.Types
open FsML.Domain.Regression
open FsML.Domain.Optimization
open FsML.Algorithms.Classification.LogisticRegression

module LogisticRegressionTests =

  type Sample = { trainingH: Matrix<float>; trainingY: Vector<float> }
 
  type TestData =
    static member Sample () =
      gen {
        let nrSamples = 100
        let neg = Gen.elements [-50.0 .. -1.0 .. -100.0] |> Gen.sample 0 (nrSamples / 2)
        let pos = Gen.elements [50.0 .. 1.0 .. 100.0] |> Gen.sample 0 (nrSamples / 2)

        let axis = (neg @ pos) |> Array.ofList

        let H = [|
                  Array.create nrSamples 1.0
                  axis
                  axis
                |] |> DenseMatrix.OfColumnArrays

        let Y = [| for i in 0 .. nrSamples - 1 do
                   if (H.[i, 1] > 0.0 && H.[i, 2] > 0.0) then yield 1.0
                   else yield 0.0 |] |> DenseVector.OfArray

        return { trainingH = H; trainingY = Y }
      } |> Arb.fromGen

  [<Property(Arbitrary=[| typeof<TestData> |])>]
  let ``Can calculate regression line using gradient descent`` ({ trainingH = H; trainingY = Y }) =
    let epsilon = 0.05

    let fit: Result<Weights, ErrorResult list> = Either.either {
      let! gdParameters = GradientDescentParameters.create GradientDescent.Stochastic 0.0001 0.05 10000u
      let! trainingParameters = TrainingParameters.create H Y
      let logisticRegressionWithRegularization = fitWithGradientDescent Regularization.Without gdParameters
      let fit = logisticRegressionWithRegularization trainingParameters
      return fit
    }

    match fit with
    | Error _ -> Assert.True(false)
    | Ok fit -> Assert.InRange(fit.At(0), -1.0 * epsilon, epsilon)
                Assert.Equal(fit.At(1), fit.At(1))


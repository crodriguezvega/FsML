namespace FsML.Tests

open FsML
open NUnit.Framework
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module LinearRegressionTests =

  let trainingX = [
                    [| 1.0; 1.0 |]
                    [| 1.0; 2.0 |]
                  ] |> DenseMatrix.OfRowArrays
  let trainingY = DenseVector.raw [| 1.0; 2.0 |]
  let theta = DenseVector.raw [| 2.0; 3.0 |]

  [<TestFixture>]
  type CalculationOfCostFunction () =

    [<Test>]
    member this.WithoutRegularization () =
      Assert.AreEqual(13.0, LinearRegression.costFunction trainingX trainingY theta (Optimization.Regularization.Without), 0.01)

    [<Test>]
    member this.WithRegularization () =
      Assert.AreEqual(35.5, LinearRegression.costFunction trainingX trainingY theta (Optimization.Regularization.With(10.0)), 0.01)

  [<TestFixture>]
  type CalculationOfGradientOfCostFunction () =

    [<Test>]
    member this.WithoutRegularization () =
      let gradient = LinearRegression.gradientOfCostFunction trainingX trainingY theta (Optimization.Regularization.Without)
      Assert.AreEqual(5.0, gradient.At(0), 0.01)
      Assert.AreEqual(8.0, gradient.At(1), 0.01)

    [<Test>]
    member this.WithRegularization () =
      let gradient = LinearRegression.gradientOfCostFunction trainingX trainingY theta (Optimization.Regularization.With(10.0))
      Assert.AreEqual(5.0, gradient.At(0), 0.01)
      Assert.AreEqual(23.0, gradient.At(1), 0.01)
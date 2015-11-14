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
  let trainingY = vector([ 1.0; 2.0 ])
  let theta = vector([ 2.0; 3.0 ])

  [<TestFixture>]
  type CalculationOfCostFunction () =

    [<Test>]
    member this.withoutRegularization () =
      let cost = LinearRegression.costFunction Optimization.Regularization.Without trainingX trainingY theta

      Assert.AreEqual(13.0, cost, 0.01)

    [<Test>]
    member this.withRegularization () =
      let cost = LinearRegression.costFunction (Optimization.Regularization.With(10.0)) trainingX trainingY theta

      Assert.AreEqual(35.5, cost, 0.01)

  [<TestFixture>]
  type CalculationOfGradientOfCostFunction () =

    [<Test>]
    member this.withoutRegularization () =
      let gradient = LinearRegression.gradientOfCostFunction Optimization.Regularization.Without Optimization.GradientDescent.Standard trainingX trainingY theta

      Assert.AreEqual(5.0, gradient.At(0), 0.01)
      Assert.AreEqual(8.0, gradient.At(1), 0.01)

    [<Test>]
    member this.withRegularization () =
      let gradient = LinearRegression.gradientOfCostFunction (Optimization.Regularization.With(10.0)) Optimization.GradientDescent.Standard trainingX trainingY theta

      Assert.AreEqual(5.0, gradient.At(0), 0.01)
      Assert.AreEqual(23.0, gradient.At(1), 0.01)
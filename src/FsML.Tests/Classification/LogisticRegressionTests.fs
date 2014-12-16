namespace FsML.Tests

open FsML
open NUnit.Framework
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module LogisticRegressionTests =

  let trainingX = [
                    [| 1.0; -5.0; -5.0 |]
                    [| 1.0; 5.0; 5.0 |]
                  ] |> DenseMatrix.OfRowArrays
  let trainingY = vector([ 0.0; 1.0 ])
  let theta = vector([ 0.0; 1.0; 1.0 ])

  [<TestFixture>]
  type CalculationOfSigmoidFunction () =

    [<Test>]
    member this.withMatrix () =
      let z = LogisticRegression.sigmoidFunction (trainingX * theta)

      Assert.AreEqual(0.0, z.At(0), 0.01)
      Assert.AreEqual(1.0, z.At(1), 0.01)

  [<TestFixture>]
  type CalculationOfCostFunction () =

    [<Test>]
    member this.withoutRegularization () =
      Assert.AreEqual(0.0, LogisticRegression.costFunction trainingX trainingY theta (Optimization.Regularization.Without), 0.01)

    [<Test>]
    member this.withRegularization () =
      Assert.AreEqual(5.0, LogisticRegression.costFunction trainingX trainingY theta (Optimization.Regularization.With(10.0)), 0.01)

  [<TestFixture>]
  type CalculationOfGradientCostFunction () =

    [<Test>]
    member this.withoutRegularization () =
      let gradient = LogisticRegression.gradientOfCostFunction trainingX trainingY theta (Optimization.Regularization.Without)

      Assert.AreEqual(0.0, gradient.At(0), 0.01)
      Assert.AreEqual(0.0, gradient.At(1), 0.01)
      Assert.AreEqual(0.0, gradient.At(2), 0.01)

    [<Test>]
    member this.withRegularization () =
      let gradient = LogisticRegression.gradientOfCostFunction trainingX trainingY theta (Optimization.Regularization.With(10.0))

      Assert.AreEqual(0.0, gradient.At(0), 0.01)
      Assert.AreEqual(5.0, gradient.At(1), 0.01)
      Assert.AreEqual(5.0, gradient.At(2), 0.01)
namespace FsML.Tests

open FsML
open NUnit.Framework
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module LinearRegressionTests =

  let trainingX = [
                    [| 1.0; 2.0 |]
                    [| 2.0; 1.0 |]
                  ] |> DenseMatrix.OfRowArrays
  let trainingY = DenseVector.raw [| 1.0; 2.0|]
  let theta = DenseVector.raw [|2.0; 1.0|]

  [<TestFixture>]
  type CalculationOfCostFunction () =

    [<Test>]
    member this.WithoutRegularization () =
      Assert.AreEqual(4.5, LinearRegression.costFunction trainingX trainingY theta (Optimization.Regularization.Without), 0.01)
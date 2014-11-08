namespace FsML.Tests

open FsML
open NUnit.Framework
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module NeuralNetworksTests =

  [<TestFixture>]
  type ForwarPropagation () =

    [<Test>]
    member this.AND () =
      let X = [|
                [| 0.0; 0.0 |]
                [| 0.0; 1.0 |]
                [| 1.0; 0.0 |]
                [| 1.0; 1.0 |]
              |] |> DenseMatrix.OfRowArrays
      let thetas: Matrix<_> list = [ [| -30.0; 20.0; 20.0 |] |> DenseMatrix.OfRowArrays ]

      let az = NeuralNetworks.forwardPropagation X thetas
      let output = fst (List.rev az).Head

      Assert.AreEqual(0.0, output.At(0, 0), 0.01)
      Assert.AreEqual(0.0, output.At(1, 0), 0.01)
      Assert.AreEqual(0.0, output.At(2, 0), 0.01)
      Assert.AreEqual(1.0, output.At(3, 0), 0.01)
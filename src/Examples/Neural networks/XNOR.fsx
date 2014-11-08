#I "../../../build/debug"
#I "../../../packages/FSharp.Charting.0.90.7/lib/net40"
#I "../../../packages/MathNet.Numerics.3.2.3/lib/net40"
#I "../../../packages/MathNet.Numerics.FSharp.3.2.3/lib/net40"

#r "FsML.dll"
#r "FSharp.Charting.dll"
#r "MathNet.Numerics.dll"
#r "MathNet.Numerics.FSharp.dll"

open System
open System.Drawing

open FsML
open FSharp.Charting
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Distributions

module XNOR =

// Each row is a sample
let X = [|
          [| 0.0; 0.0 |]
          [| 0.0; 1.0 |]
          [| 1.0; 0.0 |]
          [| 1.0; 1.0 |]
        |] |> DenseMatrix.OfRowArrays

// The length of the list determines the number of layers (hidden plus output)
// Each row of a theta matrix contains the coefficients for each unit of the layer from top to bottom
let thetas: Matrix<_> list = [
  [|
    [| -30.0; 20.0; 20.0 |]
    [| 10.0; -20.0; -20.0 |]
  |] |> DenseMatrix.OfRowArrays;
  [|
    [| -10.0; 20.0; 20.0 |]
  |] |> DenseMatrix.OfRowArrays
]

// Calculate intermediate values between layers, and also output
let az = NeuralNetworks.forwardPropagation X thetas

// Extract output of neural network
let output = fst (List.rev az).Head
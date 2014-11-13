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

module WithoutRegularization = fsi.AddPrinter(fun (ch:FSharp.Charting.ChartTypes.GenericChart) -> ch.ShowChart() |> ignore; "(Chart)")

let length = 200
let positiveContinuousDistribution = ContinuousUniform(1.0, 2.0)
let negativeContinuousDistribution = ContinuousUniform(-2.0, -1.0)
let x1 = [| for i in 1 .. length do
              if i <= 100 then yield positiveContinuousDistribution.Sample()
              else yield negativeContinuousDistribution.Sample() |]
let x2 = [| for i in 1 .. length do
              if (i <= 50 || i > 150) then yield positiveContinuousDistribution.Sample()
              else yield negativeContinuousDistribution.Sample() |]

// Each row is a training sample
let trainingX = [|
                  x1
                  x2
                |] |> DenseMatrix.OfColumnArrays

// Each row is the ouput value for each training sample
let trainingY = [| for i in 0 .. trainingX.RowCount - 1 do
                     if (trainingX.[i, 0] >= 1.0 && trainingX.[i, 1] >= 1.0) then yield [| 1.0; 0.0; 0.0; 0.0|]
                     else if (trainingX.[i, 0] >= 1.0 && trainingX.[i, 1] <= -1.0) then yield [| 0.0; 1.0; 0.0; 0.0|]
                     else if (trainingX.[i, 0] <= -1.0 && trainingX.[i, 1] <= -1.0) then yield [| 0.0; 0.0; 1.0; 0.0|]
                     else yield [| 0.0; 0.0; 0.0; 1.0|]
                |] |> DenseMatrix.OfRowArrays

// Initialize theta matrices with random values: we will use a neural network with one hidden layer
let thetas: Matrix<_> list = [ DenseMatrix.randomStandard<float> 3 3; DenseMatrix.randomStandard<float> 4 4 ]

let fit = NeuralNetworks.train trainingX trainingY thetas 0.1 2000 Optimization.Regularization.Without

// New input samples we need to classify
let X = [|
          [| 1.5; 1.5 |]    // Prediction should be close to [| 1.0; 0.0; 0.0; 0.0 |]
          [| 1.5; -1.5 |]   // Prediction should be close to [| 0.0; 1.0; 0.0; 0.0 |]
          [| -1.5; -1.5 |]  // Prediction should be close to [| 0.0; 0.0; 1.0; 0.0 |]
          [| -1.5; 1.5 |]   // Prediction should be close to [| 0.0; 0.0; 0.0; 1.0 |]
        |] |> DenseMatrix.OfRowArrays

// Calculate intermediate values between layers, and also output
let prediction = NeuralNetworks.predict X fit

// Plot training samples
let points = (x1, x2) ||> Array.map2 (fun x y -> (x, y))
Chart.Combine([ Chart.Point(points) ])
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

module WithSpecificCentroids = fsi.AddPrinter(fun (ch:FSharp.Charting.ChartTypes.GenericChart) -> ch.ShowChart() |> ignore; "(Chart)")

let length = 150

let xContinuousDistribution = ContinuousUniform(-0.25, 0.25)
let y1ContinuousDistribution = ContinuousUniform(-2.0, -1.5)
let y2ContinuousDistribution = ContinuousUniform(-0.25, 0.25)
let y3ContinuousDistribution = ContinuousUniform(1.5, 2.0)

let x = [| for i in 1 .. length -> xContinuousDistribution.Sample() |]
let y = [| for i in 1 .. length do
             if (i <= 50) then yield y1ContinuousDistribution.Sample()
             else if (i <= 100 && i > 50) then yield y2ContinuousDistribution.Sample()
             else yield y3ContinuousDistribution.Sample() |]

// Each row is a sample
let points = [|
               x
               y
             |] |> DenseMatrix.OfColumnArrays |> Matrix.toRowSeq

// Each row is an initial centroid
let initialCentroids = [|
                         [| -0.25; -2.0 |]
                         [| -0.25; -0.25 |]
                         [| -0.25; 1.5 |]
                       |] |> DenseMatrix.OfRowArrays |> Matrix.toRowSeq

// Find the the optimal centroid for each point
let optimalCentroids = Kmeans.findOptimalCentroids points initialCentroids 100
let optimalAssignment = Kmeans.cluster points optimalCentroids

Chart.Combine(
  [ Chart.Point ((x, y) ||> Array.map2 (fun x y -> (x, y)))
    Chart.Point (optimalCentroids |> Seq.map (fun x -> (x.At(0), x.At(1))) |> Array.ofSeq) ])
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

module WithPolynomialFeatures = fsi.AddPrinter(fun (ch:FSharp.Charting.ChartTypes.GenericChart) -> ch.ShowChart() |> ignore; "(Chart)")

let normalDistribution = Normal.WithMeanVariance(5.0, 10.0)
let x = [| -5.0 .. 0.5 .. 5.0 |]
let y = x |> Array.map (fun x -> x + (pown x 2) + (pown x 3) + normalDistribution.Sample())

// Each row is a training sample
let trainingX = [|
                  (Array.create x.Length 1.0) |> Array.toList |> vector
                  x |> Array.toList |> vector
                  (x |> DenseVector.OfArray).PointwisePower(2.0)
                  (x |> DenseVector.OfArray).PointwisePower(3.0)
                |] |> DenseMatrix.OfColumnVectors

// Each element is the ouput value for each training sample
let trainingY = y |> DenseVector.OfArray

let fit = LinearRegression.fitWithNormalEquation trainingX trainingY Optimization.Regularization.Without

Chart.Combine(
  [ Chart.Point ((x, y) ||> Array.map2 (fun x y -> (x, y)))
    Chart.Line (x |> Array.map (fun x -> (x, fit.At(0) + fit.At(1) * (pown x 2) + fit.At(2) * (pown x 3)))) ])
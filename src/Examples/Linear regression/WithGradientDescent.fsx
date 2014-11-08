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

module WithGradientDescent = fsi.AddPrinter(fun (ch:FSharp.Charting.ChartTypes.GenericChart) -> ch.ShowChart() |> ignore; "(Chart)")

let normalDistribution = Normal.WithMeanVariance(0.0, 2.0)
let x = [|1.0 .. 1.0 .. 10.0|]
let y = x |> Array.map (fun x -> x + normalDistribution.Sample())

// Each row is a training sample
let trainingX = x |> DenseMatrix.OfColumnArrays

// Each element is the ouput value for each training sample
let trainingY = y |> DenseVector.OfArray

let fit = LinearRegression.fitWithGradientDescent trainingX trainingY 0.01 1500 Optimization.Regularization.Without

Chart.Combine(
  [ Chart.Point ((x, y) ||> Array.map2 (fun x y -> (x, y)))
    Chart.Line (x |> Array.map (fun x -> (x, fit.At(0) + fit.At(1) * x))) ])
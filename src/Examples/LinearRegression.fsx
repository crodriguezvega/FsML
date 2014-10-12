#I "../../build/debug"
#I "../../packages/FSharp.Charting.0.90.7/lib/net40"
#I "../../packages/MathNet.Numerics.3.2.3/lib/net40"
#I "../../packages/MathNet.Numerics.FSharp.3.2.3/lib/net40"

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

module FSharpCharting = fsi.AddPrinter(fun (ch:FSharp.Charting.ChartTypes.GenericChart) -> ch.ShowChart(); "FSharpCharting")

let x = [|1.0 .. 1.0 .. 10.0|]
let y = x |> Array.map (fun x -> x + Normal.WithMeanVariance(0.0, 2.0).Sample())
let trainingX = [|Array.create x.Length 1.0; x|] |> DenseMatrix.OfColumnArrays
let trainingY = y |> DenseVector.OfArray

let fit = LinearRegression.fitWithNormalEquation trainingX trainingY

Chart.Combine(
  [ Chart.Point ((x, y) ||> Array.map2 (fun x y -> (x, y)))
    Chart.Line (x |> Array.map (fun x -> (x, fit.At(0) + fit.At(1) * x))) ])
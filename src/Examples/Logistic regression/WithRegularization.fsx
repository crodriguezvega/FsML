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

module WithRegularization = fsi.AddPrinter(fun (ch:FSharp.Charting.ChartTypes.GenericChart) -> ch.ShowChart() |> ignore; "(Chart)")

let length = 100
let normalDistribution = Normal.WithMeanVariance(0.0, 2.0)
let x1 = [| for i in 1 .. 100 -> normalDistribution.Sample() |]
let x2 = [| for i in 1 .. 100 -> normalDistribution.Sample() |]

// Each row is a training sample
let trainingX = [|
                  (Array.create length 1.0) |> Array.toList |> vector
                  (x1 |> DenseVector.OfArray).PointwisePower(2.0)
                  x2 |> Array.toList |> vector
                |] |> DenseMatrix.OfColumnVectors

// Each element is the ouput value for each training sample
let trainingY = [| for i in 0 .. length - 1 do
                   if (trainingX.[i, 2] >= (pown trainingX.[i, 1] 2)) then yield 1.0
                   else yield 0.0 |] |> DenseVector.OfArray

let logisticRegressionWithRegularization = LogisticRegression.fitWithGradientDescent (Optimization.Regularization.With(5.0)) Optimization.GradientDescent.Standard
let fit = logisticRegressionWithRegularization trainingX trainingY 0.05 10000

let points = (x1, x2) ||> Array.map2 (fun x y -> (x, y))
Chart.Combine(
  [ Chart.Point(points |> Array.filter (fun x -> (snd x < pown (fst x) 2)))
    Chart.Point(points |> Array.filter (fun x -> (snd x >= pown (fst x) 2)))
    Chart.Line([|(Array.min x1 - 0.1) .. 0.1 .. (Array.max x1 + 0.1)|] |> Array.map (fun x -> x, -1.0 *(fit.At(0) + fit.At(1) * (pown x 2)) / fit.At(2)))
         .WithXAxis(MajorTickMark = ChartTypes.TickMark(Interval = 1.0)) ])
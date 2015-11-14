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

let length = 100
let normalDistribution = Normal.WithMeanVariance(0.0, 2.0)
let x1 = [| for i in 1 .. length -> normalDistribution.Sample() |]
let x2 = [| for i in 1 .. length -> normalDistribution.Sample() |]

// Each row is a training sample
let trainingX = [|
                  Array.create length 1.0
                  x1
                  x2
                |] |> DenseMatrix.OfColumnArrays

// Each element is the ouput value for each training sample
let trainingY = [| for i in 0 .. length - 1 do
                   if (trainingX.[i, 1] + trainingX.[i, 2]) >= 1.0 then yield 1.0
                   else yield 0.0 |] |> DenseVector.OfArray

let logisticRegressionWithoutRegularization = LogisticRegression.fitWithGradientDescent Optimization.Regularization.Without Optimization.GradientDescent.Standard
let fit = logisticRegressionWithoutRegularization trainingX trainingY 0.05 5000 

// Predict output (it should be 1)
let input = matrix([ [ 1.0; 3.0; 3.0 ] ])
let prediction = LogisticRegression.predict input fit

let points = (x1, x2) ||> Array.map2 (fun x y -> (x, y))
Chart.Combine(
  [ Chart.Point(points |> Array.filter (fun x -> (fst x + snd x) < 1.0))
    Chart.Point(points |> Array.filter (fun x -> (fst x + snd x) >= 1.0))
    Chart.Line([|(Array.min x1 - 0.1) .. 0.5 .. (Array.max x1 + 0.1)|] |> Array.map (fun x -> x, -1.0 *(fit.At(0) + fit.At(1) * x) / fit.At(2)))
         .WithXAxis(MajorTickMark = ChartTypes.TickMark(Interval = 1.0)) ])
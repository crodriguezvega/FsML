#I "../../../build/Debug/FsML.Algorithms"
#I "../../../build/Debug/FsML.Utilities"
#I "../../../packages/FSharp.Charting.0.91.1/lib/net45"
#I "../../../packages/MathNet.Numerics.4.4.1/lib/net461"
#I "../../../packages/MathNet.Numerics.FSharp.4.4.1/lib/net45"

#r "FsML.Algorithms.dll"
#r "FsML.Utilities.dll"
#r "FSharp.Charting.dll"
#r "MathNet.Numerics.dll"
#r "MathNet.Numerics.FSharp.dll"
#r "System.Windows.Forms.DataVisualization.dll"

open FSharp.Charting
open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

open FsML.Algorithms
open FsML.Algorithms.Regression
open FsML.Utilities.Types
open FsML.Utilities.Builders

module WithNormalEquation =

    let normalDistribution = Normal.WithMeanVariance(0.0, 5.0)
    let x = [| 1.0 .. 1.0 .. 10.0 |]
    let y = x |> Array.map (fun x -> (pown x 2) + normalDistribution.Sample())

    // Each row is a training sample
    let trainingX = [|
                        (Array.create x.Length 1.0) |> Array.toList |> vector
                        (x |> DenseVector.OfArray).PointwisePower(2.0)
                    |] |> DenseMatrix.OfColumnVectors

    // Each element is the ouput value for each training sample
    let trainingY = y |> DenseVector.OfArray

    let fit : Result<Vector<float>, ErrorResult> = Either.either {
        let! fit = LinearRegression.fitWithNormalEquation Optimization.Regularization.Without trainingX trainingY
        return fit
    }

    match fit with
    | Error e -> printfn "No fit: %A" e
    | Ok fit -> let chart = Chart.Combine([ Chart.Point ((x, y) ||> Array.map2 (fun x y -> (x, y)))
                                            Chart.Line (x |> Array.map (fun x -> (x, fit.At(0) + fit.At(1) * (pown x 2)))) ])
                chart.ShowChart() |> ignore
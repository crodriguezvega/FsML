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
open FsML.Algorithms.Optimization
open FsML.Utilities.Types
open FsML.Utilities.Builders

module WithGradientDescent =

    let normalDistribution = Normal.WithMeanVariance(0.0, 2.0)
    let x = [| 1.0 .. 1.0 .. 10.0 |]
    let y = x |> Array.map (fun x -> x + normalDistribution.Sample())

    // Each row is a training sample
    let trainingX = [|
                        Array.create x.Length 1.0 // Add intercept term
                        x
                    |] |> DenseMatrix.OfColumnArrays

    // Each element is the ouput value for each training sample
    let trainingY = y |> DenseVector.OfArray

    let fit : Result<Vector<float>, ErrorResult> = Either.either {
        let gdParameters = { category = Optimization.GradientDescent.Batch; learningRate = 0.01; numberOfiterations = 1500u }
        let linearRegressionWithBGD = LinearRegression.fitWithGradientDescent Optimization.Regularization.Without gdParameters
        let! fit = linearRegressionWithBGD trainingX trainingY
        return fit
    }

    match fit with
    | Error e -> printfn "No fit: %A" e
    | Ok fit -> let chart = Chart.Combine([ Chart.Point ((x, y) ||> Array.map2 (fun x y -> (x, y)))
                                            Chart.Line (x |> Array.map (fun x -> (x, fit.At(0) + fit.At(1) * x))) ])
                chart.ShowChart() |> ignore
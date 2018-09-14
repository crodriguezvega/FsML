namespace FsML.Common

open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Statistics

module Utils =

    let scaling (X: Vector<float>) =
        let mean = Statistics.Mean X
        let std = Statistics.StandardDeviation X
        match std with
        | 0.0 -> None
        | _ -> Some (X.Subtract(mean) / std)

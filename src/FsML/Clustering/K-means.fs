namespace FsML

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module Kmeans =

  /// Cost function
  let costFunction assignmentsPointCentroid =
    let numberOfSamples = Seq.length assignmentsPointCentroid
    let sumOfSquareDistances = Seq.fold (fun acc (elem: Vector<float> * Vector<float>) -> acc + (Distance.Euclidean(fst elem, snd elem) ** 2.0)) 0.0 assignmentsPointCentroid
    (1.0 / float numberOfSamples) * sumOfSquareDistances

  /// Calculate distances to centroids (returns seq<(point, seq<(centroid, distanceToCentroid)>)>)
  let calculateDistanceToCentroid (points: seq<Vector<float>>) (centroids: seq<Vector<float>>) =
    seq { for point in points ->
          (point, Seq.map (fun centroid -> (centroid, Distance.Euclidean(centroid, point))) centroids) }

  /// Find closest centroids (returns seq<(point, centroid)>)
  let findClosestCentroid pointsWithDistancesToCentroids =
    seq { for x in pointsWithDistancesToCentroids ->
          (fst x, (Seq.minBy (fun y -> snd y) (snd x)) |> (fun z -> fst z)) }

  /// Group points by closest centroid (returns seq<(centroid, seq<point>)>)
  let groupPointsByClosestCentroid pointsWithClosestCentroid =
    Seq.groupBy (fun x -> snd x) pointsWithClosestCentroid |> Seq.map (fun x -> (fst x, Seq.map (fun y -> fst y) (snd x)))

  /// Move centroids (returns a seq<(centroid, seq<point>)>)
  let rec moveCentroids pointsGroupedByCentroid =
    match pointsGroupedByCentroid with
    | [] -> []
    | head :: tail -> 
      let pointsAssignedToCentroid = snd head
      let numberOfPoints = Seq.length pointsAssignedToCentroid
      let numberOfDimensions = Vector.length (Seq.head pointsAssignedToCentroid)
      let zeroVector = (Array.create numberOfDimensions 0.0 |> Array.toList|> vector)
      let centroid = (Seq.fold (fun acc point -> acc + point) zeroVector pointsAssignedToCentroid).Divide(float numberOfPoints)
      centroid :: (moveCentroids tail)

  /// Find optimal centroids
  let rec findOptimalCentroids points centroids numberOfiterations =
    match numberOfiterations with
    | 0 -> centroids
    | _ ->
      let updatedCentroids = (points, centroids) ||> calculateDistanceToCentroid
                                                  |> findClosestCentroid
                                                  |> groupPointsByClosestCentroid 
                                                  |> List.ofSeq
                                                  |> moveCentroids
      findOptimalCentroids points updatedCentroids (numberOfiterations - 1)

  /// Cluster
  let rec cluster points centroids =
    (points, centroids) ||> calculateDistanceToCentroid
                         |> findClosestCentroid
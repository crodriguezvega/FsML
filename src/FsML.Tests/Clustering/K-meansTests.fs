namespace FsML.Tests

open FsML
open NUnit.Framework
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

module KmeansTests =

  [<TestFixture>]
  type CalculateDistance () =

    [<Test>]
    member this.distanceFromPointToCentroid () =
      let points = seq[ vector([ 1.0; 1.0 ]) ]
      let centroids = seq[ vector([ 2.0; 2.0 ]) ] 

      let distances = (points, centroids) ||> Kmeans.calculateDistanceToCentroid

      let centroidAndDistance = Seq.head (snd (Seq.head distances))
      let centroid = fst centroidAndDistance
      let distance = snd centroidAndDistance

      Assert.AreEqual(2.0, centroid.At(0), 0.01)
      Assert.AreEqual(2.0, centroid.At(1), 0.01)
      Assert.AreEqual(sqrt(2.0), distance, 0.01)

  [<TestFixture>]
  type ClosestCentroid () =

    [<Test>]
    member this.findClosestCentroid () =
      let point = [| 1.0; 1.0|] |> Array.toList |> vector
      let distances = seq [ (point, seq [ (vector([ 0.0; 1.0 ]), 1.0); (vector([ 1.0; 3.0 ]), 2.0)]) ]

      let closestCentroids = Kmeans.findClosestCentroid distances

      let closestCentroid = Seq.head closestCentroids
      let centroid = snd closestCentroid

      Assert.AreEqual(0.0, centroid.At(0), 0.01)
      Assert.AreEqual(1.0, centroid.At(1), 0.01)

  [<TestFixture>]
  type GroupPoints () =

    [<Test>]
    member this.groupPointsByClosestCentroid () =
      let point1 = vector([ 1.0; 1.0 ])
      let point2 = vector([ -1.0; -1.0 ])
      let centroid = vector([ 0.5; 0.5 ])
      let distances = seq [ (point1, centroid); (point2, centroid) ]

      let groups = Kmeans.groupPointsByClosestCentroid distances

      let group = Seq.head groups
      let centroid = fst group
      let numberOfPoints = Seq.length (snd group)

      Assert.AreEqual(0.5, centroid.At(0), 0.01)
      Assert.AreEqual(0.5, centroid.At(1), 0.01)
      Assert.AreEqual(2, numberOfPoints)

  [<TestFixture>]
  type UpdateCentroids () =

    [<Test>]
    member this.moveCentroids () =
      let point1 = vector([ 2.0; 2.0 ])
      let point2 = vector([ -1.0; -1.0 ])
      let centroid = vector([ 0.0; 0.0 ])
      let groupedPoints = [ (centroid, seq [ point1; point2 ]) ]

      let centroids = Kmeans.moveCentroids groupedPoints

      let centroid = (Seq.head centroids)

      Assert.AreEqual(0.5, centroid.At(0), 0.01)
      Assert.AreEqual(0.5, centroid.At(1), 0.01)

  [<TestFixture>]
  type CalculateCost () =

    [<Test>]
    member this.costFunction () =
      let point1 = vector([ 2.0; 2.0 ])
      let point2 = vector([ -1.0; -1.0 ])
      let centroid = vector([ 0.0; 0.0 ])
      let assignments = seq [ (point1, centroid); (point2, centroid) ]

      let cost = Kmeans.costFunction assignments

      Assert.AreEqual(5.0, cost , 0.01)
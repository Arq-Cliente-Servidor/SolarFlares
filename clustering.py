from __future__ import print_function

import sys
import numpy as np

from time import time, strftime
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.clustering import BisectingKMeans, BisectingKMeansModel

def parse_date(line):
    date = line[0].split('-')
    date = "-".join(date[:2])
    explosion_value = float(line[1])
    return (date, [explosion_value, 1])

def group_days(day1, day2):
    day = [day1[0] + day2[0]]
    count = [day1[1] + day2[1]]
    return day + count

def parse_month(line):
    date = line[0].split('-')
    date = date[1]
    return (date, float(line[1]))

def average_year(lines):
    rdd = lines.map(lambda line: line.split(',')) \
                .map(parse_date) \
                .reduceByKey(group_days) \
                .sortByKey() \
                .map(lambda line: (line[0], line[1][0] / line[1][1]))
    return rdd

def average_month(years):
    rdd = years.map(parse_month) \
                .reduceByKey(lambda month1, month2: month1 + month2) \
                .sortByKey() \
                .map(lambda line: line[1] / 2.0)
    return rdd

def parseDataset(lines):
    rdd = lines.map(lambda line: line.split(',')[1]) \
                .map(lambda line: [float(line)])
    return rdd

def generate_initial_centroids(months, k):
    if k == 12:
        return months
    else:
        if k == 4:
            months = [months[11]] + months[:11]
        centroids = []
        offset = 12 / k
        for i in range(0, len(months), offset):
            centroids.append(months[i : i + offset])

        centroids = map(lambda items: sum(items) / float(offset), centroids)
        centroids = map(lambda item: [item], centroids)

        return centroids


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: kmeans <k>(2, 4, 12)", file=sys.stderr)
        exit(-1)

    currTime = strftime("%x") + '-' + strftime("%X")
    currTime = currTime.replace('/', '-')
    currTime = currTime.replace(':', '-')

    sc = SparkContext(appName="KMeans")
    lines = sc.textFile("hdfs://localhost:9000/user/sebastian/dataset_observatory/initial_centroids.csv")
    dataset = sc.textFile("hdfs://localhost:9000/user/sebastian/dataset_observatory/training_data2.csv")

    average_per_year = average_year(lines) # 2014 and 2015
    average_per_month = average_month(average_per_year)
    data = parseDataset(dataset)
    k = int(sys.argv[1])
    initial_centroids = generate_initial_centroids(average_per_month.collect(), k)

    # KMeans
    start = time()
    kmeans_model = KMeans.train(data, k, maxIterations = 100, initialModel = KMeansModel(initial_centroids))
    end = time()
    elapsed_time = end - start
    kmeans_output = [
        "====================== KMeans ====================\n",
        "Final centers: " + str(kmeans_model.clusterCenters),
        "Total Cost: " + str(kmeans_model.computeCost(data)),
        "Value of K: " + str(k),
        "Elapsed time: %0.10f seconds." % elapsed_time
    ]

    # Bisecting KMeans
    start = time()
    bisecting_model = BisectingKMeans.train(data, k, maxIterations = 20,
                            minDivisibleClusterSize = 1.0, seed = -1888008604)
    end = time()
    elapsed_time = end - start
    bisecting_output = [
        "====================== Bisecting KMeans ====================\n",
        "Final centers: " + str(bisecting_model.clusterCenters),
        "Total Cost: " + str(bisecting_model.computeCost(data)),
        "Value of K: " + str(k),
        "Elapsed time: %0.10f seconds." % elapsed_time
    ]

    kmeans_info = sc.parallelize(kmeans_output)
    bisecting_info = sc.parallelize(bisecting_output)
    kmeans_info.saveAsTextFile("hdfs://localhost:9000/user/sebastian/output/kmeans_" + currTime)
    bisecting_info.saveAsTextFile("hdfs://localhost:9000/user/sebastian/output/bisecting_kmeans_" + currTime)
    sc.stop()

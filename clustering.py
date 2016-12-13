from __future__ import print_function

import sys
import numpy as np

from time import time, strftime
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.clustering import BisectingKMeans, BisectingKMeansModel

def parseDate(line):
    date = line[0].split('-')
    date = "-".join(date[:2])
    explosion_value = float(line[1])
    return (date, [explosion_value, 1])

def groupDays(day1, day2):
    day = [day1[0] + day2[0]]
    count = [day1[1] + day2[1]]
    return day + count

def parseMonth(line):
    date = line[0].split('-')
    date = date[1]
    return (date, float(line[1]))

def average_year(lines):
    rdd = lines.map(lambda line: line.split(',')) \
                .map(parseDate) \
                .reduceByKey(groupDays) \
                .sortByKey() \
                .map(lambda line: (line[0], line[1][0] / line[1][1]))
    return rdd

def average_month(years):
    rdd = years.map(parseMonth) \
                .reduceByKey(lambda month1, month2: month1 + month2) \
                .sortByKey() \
                .map(lambda line: [line[1] / 2.0])
    return rdd

def parseDataset(lines):
    rdd = lines.map(lambda line: line.split(',')[1]) \
                .map(lambda line: [float(line)])
    return rdd

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: kmeans <k> <maxIterations> ", file=sys.stderr)
    #     exit(-1)

    currTime = strftime("%x") + '-' + strftime("%X")
    currTime = currTime.replace('/', '-')
    currTime = currTime.replace(':', '-')

    sc = SparkContext(appName="KMeans")
    lines = sc.textFile("hdfs://localhost:9000/user/sebastian/dataset_observatory/training_data.csv")

    average_per_year = average_year(lines) # 2014 and 2015
    average_per_month = average_month(average_per_year)
    data = parseDataset(lines)
    initial_centroids = average_per_month.collect()

    # KMeans
    start = time()
    k = 12
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
                            minDivisibleClusterSize=1.0, seed=-1888008604)
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

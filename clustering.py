from __future__ import print_function

import sys
import numpy as np

from random import random
from time import time
from os import path
from subprocess import call, CalledProcessError
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel

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

    sc = SparkContext(appName="KMeans")
    lines = sc.textFile("hdfs://localhost:9000/user/sebastian/dataset_observatory/training_data.csv")

    average_per_year = average_year(lines) # 2014 and 2015
    average_per_month = average_month(average_per_year)
    data = parseDataset(lines)

    start = time()
    k = 12
    model = KMeans.train(data, k, maxIterations = 100, initialModel = KMeansModel(average_per_month.collect()))
    end = time()
    elapsed_time = end - start
    output = [
        "Final centers: " + str(model.clusterCenters),
        "Total Cost: " + str(model.computeCost(data)),
        "Value of K: " + str(k),
        "Elapsed time: %0.10f seconds." % elapsed_time
    ]

    info = sc.parallelize(output)
    info.saveAsTextFile("hdfs://localhost:9000/user/sebastian/output")
    sc.stop()

from __future__ import print_function

import sys
import numpy as np
import random
import time
import threading
import random

import socket

from pyspark import SparkContext

from pycuda import gpuarray
from pycuda.compiler import SourceModule
import pycuda.driver as cuda


cudakernel = """
__global__ void assignToCentroid(float *data, float *kPoints,
                                 int size, int K, float2* clusters){
    unsigned idx = blockIdx.x * blockDim.x * blockDim.y +
                   threadIdx.y * blockDim.x + threadIdx.x;

    if (idx < size) {
        float closest = INFINITY;
        float temp_dist = 0;
        for (int i = 0; i < K; i++) {
            temp_dist = fabsf(kPoints[i] - data[idx]);
            if (temp_dist < closest) {
                closest = temp_dist;
                clusters[idx].x = i;
                clusters[idx].y = data[idx];
            }
        }
    }
}
"""


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
    rdd = lines.map(lambda line: [float(line)])
    return rdd


def generate_initial_centroids(months, k):
    if k == 12:
        centroids = map(lambda item: [item], months)
    else:
        if k == 4:
            months = [months[11]] + months[:11]
        centroids = []
        offset = 12 / k
        for i in range(0, len(months), offset):
            centroids.append(months[i: i + offset])
        centroids = map(lambda items: sum(items) / float(offset), centroids)
        centroids = map(lambda item: [item], centroids)
    return np.asarray(centroids, dtype=np.float32)


def gpuKMeans(dev_id, rdd, kPoints):
    print(type(rdd), type(kPoints))

    def gpuFunc(iterator):
        iterator = iter(iterator)
        cpu_data = np.asarray(list(iterator), dtype=np.float32)
        datasize = len(cpu_data)
        # * 3 for data dimensions. /256 for block size.
        gridNum = int(np.ceil(datasize / 256.0))

        # +1 for overprovisioning in case there is dangling threads
        centroids = np.empty(datasize, gpuarray.vec.float2)

        cuda.init()
        dev = cuda.Device(dev_id)
        contx = dev.make_context()

        # The GPU kernel below takes centroids IDs and 1-D data points in
        # form of float2 (x,y). X is for the centroid ID whereas Y is
        # the actual point coordinate.
        try:
            mod = SourceModule(cudakernel)
            func = mod.get_function("assignToCentroid")
            func(cuda.In(cpu_data), cuda.In(kPoints), np.int32(datasize),
                 np.int32(len(kPoints)), cuda.Out(centroids),
                 block=(16, 16, 1), grid=(gridNum, 1), shared=0)
            closest = [(val[0], (np.asarray(val[1]), 1)) for val in centroids]
        except Exception as err:
            raise Exception("Error {} in node {}".format(err,
                                                         socket.gethostname()))
        contx.pop()
        del cpu_data
        del datasize
        del centroids
        del contx
        return iter(closest)

    vals = rdd.mapPartitions(gpuFunc)
    return vals


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print >> sys.stderr, "Usage: ./spark-submit k(2, 4, 12) error_min"
        exit(-1)

    start = time.time()

    sc = SparkContext(appName="CuKMeans")

    centroids = sc.textFile(
        "hdfs://masterNode:9000/user/spark/dataset_observatory/initial_centroids.csv")
    dataset = sc.textFile(
        "hdfs://masterNode:9000/user/spark/dataset_observatory/training_data.csv")

    average_per_year = average_year(centroids)
    average_per_month = average_month(average_per_year)
    data = parseDataset(dataset)

    k = int(sys.argv[1])
    kPoints = generate_initial_centroids(average_per_month.collect(), k)
    print("Centroids:\n{}".format(kPoints))

    error_min = float(sys.argv[2])
    temp_error = float("+inf")

    while temp_error > error_min:
        # 1. Assign points to centroids parallely using GPU
        # data = sc.parallelize(data.takeSample(False, 100))
        print("Data Size:", data.count())
        closest = gpuKMeans(0, temp, kPoints)

        print("closest")
        for (x, y) in closest.collect():
            print(x, y)

        # 2. Update the  centroids on CPU
        # TODO: move pointStats calculation on GPU
        pointStats = closest.reduceByKey(
            lambda (x1, y1), (x2, y2): (x1 + x2, y1 + y2))

        print("pointStats")
        for (x, y) in pointStats.collect():
            print(x, y)

        newPoints = pointStats.map(
            lambda (x, (y, z)): (x, y / z)).collect()

        print("newPoints")
        for (x, y) in newPoints:
            print(x, y)

        # Update distance on CPU
        temp_error = sum(np.sum((kPoints[int(x)] - y) ** 2)
                         for (x, y) in newPoints) / k

        for (x, y) in newPoints:
            kPoints[x] = y

    print("Final centers:", str(kPoints))

    sc.stop()
    stop = time.time()

print("Completion time:", (stop - start), "seconds")

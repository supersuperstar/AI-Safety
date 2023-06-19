from GlobalConfig import *
from Utils import *
from Detection.Detection import Detector
from DataSet.DataSet import MyDataset, create_dataLoader, create_dataset_for_each_label,sort_dataset_by_label
from sklearn.cluster import KMeans
import logging
import numpy as np
import cv2
from tqdm import tqdm
from scipy import stats


class MyKMeans(Detector):

    def __init__(self, dataset, name='KMeans', model=None, device=None, myLogger=Detector.loggerDetector,
                 myLoggerLevel=LOGGING_LEVEL_DEBUG, myLoggerFormat=LOGGING_FORMAT, myLoggerPath=FILENAME_DEFAULT_LOG,
                 **kwargs):
        super().__init__(name, dataset, model=model, device=device, myLogger=myLogger, myLoggerLevel=myLoggerLevel,
                         myLoggerFormat=myLoggerFormat, myLoggerPath=myLoggerPath)
        self._parseArgs(**kwargs)

    def _parseArgs(self, **kwargs):
        pass

    def detect(self):
        self._loadData()
        self.labels_list = []
        totalNum = len(self.dataset.data)
        noiseNum = 0
        self.logger.info(f'{self.name}开始检测')
        # self._kMeans(self.dataset, cluster=len(list(set(self.dataset.labels))))
        # self.predict_labels = self.kmeans.labels_
        for i in tqdm(range(len(self.dataset_list))):
            self.labels_list.append(self._kMeans(self.dataset_list[i]))
            noiseNum += self.labels_list[i].sum()
        # detect if the dataset is poisoned by the num of True in labels_list
        # self._show_distance_histogram(self.distances)
        self.anormal_rate = noiseNum / totalNum
        self.logger.info(f'{self.name}检测完成')
        self.logger.info(f'{self.name}检测结果：异常样本数{noiseNum}，异常率{self.anormal_rate*100:.4f}%')
        if (self.anormal_rate > 0.005):
            self.logger.info(f'{self.name}检测结果：异常率超出阈值，数据集存在被攻击风险')
        else:
            self.logger.info(f'{self.name}检测结果：异常率处于正常范围，数据集未被攻击')
        return self.anormal_rate

    def _kMeans(self, dataset: MyDataset, cluster=1):
        numpy_data = dataset.data.reshape(len(dataset), -1)
        # Initialize KMeans
        self.kmeans = KMeans(n_clusters=cluster, n_init=10, random_state=0)
        # Fit the KMeans model to your data
        self.kmeans.fit(numpy_data)
        if(cluster == 1):
            self.distances = self.kmeans.transform(numpy_data).flatten()
        else:
            self.distances = self.kmeans.transform(numpy_data)
        # data = np.array([self.distances]).flatten()
        # # 使用 boxcox 转换数据，lambda_ 是最优的 lambda 值
        # data_boxcox, lambda_ = stats.boxcox(data)
        # # 使用 z-score 标准化来将数据转换成正态分布
        # data_norm = (data_boxcox - data_boxcox.mean()) / data_boxcox.std()
        # stats.probplot(data_norm, dist="norm", plot=plt)
        # plt.show()
        # sorted_dataset = self._sort_by_distance(numpy_data, dataset)
        # distance = np.zeros((len(dataset), len(dataset)))
        # mean_distance = np.zeros(len(dataset))
        # for i in tqdm(range(len(numpy_data))):
        #     x = numpy_data[i]
        #     for j in range(i, len(numpy_data)):
        #         y = numpy_data[j]
        #         distance[i][j]=np.linalg.norm(x-y)
        #         distance[j][i]=distance[i][j]
        #     mean_distance[i]=np.mean(distance[i])
        # self._show_distance_histogram()
        # Split the distances into 100 bins and count the number of data points in each bin
        # counts, bin_edges = np.histogram(mean_distance, bins=100)
        # # Find the bin with the most data points
        # max_bin_index = np.argmax(counts)
        # Calculate the midpoint of the bin as the threshold
        threshold = 1.24 * np.mean(self.distances)
        noise = self.distances > threshold
        return noise

    def _loadData(self):
        self.logger.info(f'{self.name}开始加载数据')
        transform = None
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
        self.dataset_list = create_dataset_for_each_label(self.dataset, transform=transform)
        # self.dataset = sort_dataset_by_label(self.dataset, transform=transform)
        self.logger.info(f'{self.name}加载数据完成')

    def _sort_by_distance(self, numpy_data, dataset):
        data_with_distances = list(zip(numpy_data.reshape(len(dataset), 28, 28), self.distances.flatten()))
        # Sort the list of tuples by distance, in descending order
        data_sorted_by_distance = sorted(data_with_distances, key=lambda x: x[1], reverse=True)
        # Extract the sorted data
        sorted_data = np.array([data for data, distance in data_sorted_by_distance])
        sorted_dataset = MyDataset(sorted_data, dataset.labels, dataset.transform)
        return sorted_dataset

    def _show_distance_picture(self, numpy_data, dataset):
        sorted_dataset = self._sort_by_distance(numpy_data, dataset)
        show_mnist(sorted_dataset, 10, 2, False)

    def _show_distance_histogram(self, distance):
        distances_flattened = distance  # Make sure it is a 1-D array
        plt.hist(distances_flattened, bins=500)  # Adjust the number of bins as needed
        plt.title('Histogram of Distances')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.show()
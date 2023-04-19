import json
import urllib
import numpy as np
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal

class SolarDatasetLoader(object):
    """Hourly solar radiation of observatories from South Korean  for 2 years. 
    Vertices represent 44 cities and the weighted edges represent the strength of the relationship. 
    The target variable allows regression operations. 
    (The weight is the correlation coefficient of solar radiation by region.)
    """

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/pinkocto/noteda/main/posts/SOLAR/data3/stgcn_data1.json"
        self._dataset = json.loads(urllib.request.urlopen(url).read().decode())

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        stacked_target = np.stack(self._dataset["FX"])
        self.features = [
            stacked_target[i : i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]
        self.targets = [
            stacked_target[i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]

    def get_dataset(self, lags: int = 4) -> StaticGraphTemporalSignal:
        """Returning the Windmill Output data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Windmill Output dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
    
    
class SolarEPTDatasetLoader(object):
    """Hourly solar radiation of observatories from South Korean  for 2 years. 
    Vertices represent 44 cities and the weighted edges represent the strength of the relationship. 
    The target variable allows regression operations. 
    The weight is the EPT correlation coefficient of solar radiation by region.
    """

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/pinkocto/noteda/main/posts/SOLAR/data3/stgcn_data2.json"
        self._dataset = json.loads(urllib.request.urlopen(url).read().decode())

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        stacked_target = np.stack(self._dataset["FX"])
        self.features = [
            stacked_target[i : i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]
        self.targets = [
            stacked_target[i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]

    def get_dataset(self, lags: int = 4) -> StaticGraphTemporalSignal:
        """Returning the Windmill Output data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Windmill Output dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
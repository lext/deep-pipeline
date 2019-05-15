import datetime
import pickle


class GlobalKVS(object):
    """
    Singleton dictionary with timestamps.
    Handy tool for data exchange in Machine Learning pipelines.

    """
    _instance = None
    _d = dict()
    _save_dir = None

    def __new__(cls, save_dir=None):
        if not cls._instance:
            cls._instance = super(GlobalKVS, cls).__new__(cls)
        if cls._instance._save_dir is None:
            cls._instance._save_dir = save_dir
        return cls._instance

    def set_save_dir(self, save_dir=None):
        self._save_dir = save_dir

    def update(self, tag, value, dtype=None, save_state=True):
        """
        Updates the internal state of the logger.

        Parameters
        ----------
        tag : str
            Tag, of the variable, which we log.
        value : object
            The value to be logged
        dtype :
            Container which is used to store the values under the tag
        save_state : bool
            Whether to pickle state
        Returns
        -------

        """
        if tag not in self._d:
            if dtype is None:
                self._d[tag] = (value, str(datetime.datetime.now()))
            else:
                self._d[tag] = dtype()
        else:
            if isinstance(self._d[tag], list):
                self._d[tag].append((value, str(datetime.datetime.now())))
            elif isinstance(self._d[tag], dict):
                self._d[tag].update((value, str(datetime.datetime.now())))
            else:
                self._d[tag] = (value, str(datetime.datetime.now()))
        if save_state:
            if self._save_dir is not None:
                self.save_pkl(self._save_dir)

    def __getitem__(self, tag):
        if not isinstance(self._d[tag], (list, dict)):
            return self._d[tag][0]
        else:
            return self._d[tag]

    def tag_ts(self, tag):
        return self._d[tag]

    def __contains__(self, key):
        return key in self._d

    def save_pkl(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self._d, f)
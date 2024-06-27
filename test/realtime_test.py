import numpy as np

from absl.testing import absltest

from telluride_decoding import realtime


class RealTimeTest(absltest.TestCase):

  def test_datastream(self):
    ds = realtime.DataStream(6, int)
    b = np.reshape(np.arange(8), (4, 2))
    ds.add_data(b)
    np.testing.assert_equal(ds._data,
                            np.array([[0, 1], [2, 3], [4, 5], 
                                      [6, 7], [0, 0], [0, 0]]))
    self.assertFalse(ds.get_data(4, 2))

    ds.add_data(b+8)
    np.testing.assert_equal(ds._data, 
                            np.asarray([[12, 13], [14, 15], [ 4,  5], 
                                        [ 6,  7], [ 8,  9], [10, 11]]))

    d = ds.get_data(5, 4)
    np.testing.assert_equal(d, np.asarray([[10, 11], [12, 13], [14, 15]]))

    self.assertFalse(ds.get_data(16, 4))

    ground_truth = np.concatenate((b, b+8), axis=0)
    for i in range(2, 10):
      trial = ds.get_data(i, 4)
      np.testing.assert_equal(ground_truth[i:i+4, :], trial)


if __name__ == '__main__':
  absltest.main()
"""此模块是测试盒 testcases，用于测试 average_precision_metric 内的函数和类。"""

import unittest

import numpy as np
import tensorflow as tf

import average_precision_metric


# @unittest.skip  # 测试通过后，可以设置为跳过，不再运行该测试盒 testcase。
class TestMeanAveragePrecision(unittest.TestCase):
    """测试 average_precision_metric 中的指标类 MeanAveragePrecision.

    使用说明：使用该测试盒时，必须把类 MeanAveragePrecision 中的方法 update_state 设
    置为 use_predictor=False。因为该测试盒的 y_pred 是已经转换为概率后的结果，无
    须用 predictor 再次转换。

    测试 13 种情况：
    1. 只有 1 张图片，只有 1 个类别，标签和预测结果完全相同，此时 AP 应该为 1。
    2. 只有 1 张图片，有 2 个类别，标签和预测结果完全相同，此时 AP 应该为 1。
    3. 只有 1 张图片，只有 1 个类别，预测结果的 IoU 为 0.64，此时 AP 应该为 0.3。
    4. 只有 1 张图片，只有 1 个类别，预测结果的 IoU 为 0.49，此时 AP 应该为 0。
    5.1 只有 1 张图片，只有 1 个类别，预测结果的 objectness 为 0.49，此时 AP 应该为 0。
    5.2 只有 1 张图片，只有 1 个类别，但是有 2 个预测结果。第一个正确预测结果 bbox
        和标签完全相同，第二个错误预测结果（因为标签中没有这个物体，所以是预测结果错误）
        bbox，objectness 和 classification 都为 0.51，此时 AP 应该为 0.75。
    6. 只有 1 张图片，只有 1 个类别，预测结果的类别置信度为 0.49，此时 AP 应该为 0。
    7. 有 2 张图片，只有 1 个类别，标签和预测结果完全相同，此时 AP 应该为 1。
    8. 有 2 张图片，只有 1 个类别，一张图片预测结果的 IoU 为 0.49，则其 AP 将为 0，另一
        张图片 IoU 为 1，此时最终 AP 应为 0.375。
    9. 有 2 张图片，只有 1 个类别。预测结果的一个 bbox 和标签完全相同，另一个 objectness
        为 0.49，此时最终 AP 应为 0.5。
    10. 有 2 张图片，只有 1 个类别，一张图片的类别置信度为 0.49，则其 AP 将为 0，此时
        最终 AP 应为 0.5。
    11. 有 2 个类别，每个类别都有 2 张图片。一个类别的 AP 为 0.375，另一个类别
        的 AP 为 1，最终 AP 应该为 0.6875。
    12. 测试指标清零后的状态。注意该测试会把 average_precision_metric 中的 3 个状态量清零。
    """

    def setUp(self):
        """将类 MeanAveragePrecision 个体化 instantiate，然后该个体就可以被用于各个
        单元测试。"""

        # 如果类 MeanAveragePrecision 是放在 yolo_v4_csp 文件中，则要用 yolo_v4_csp
        # 文件代替下面的 average_precision_metric。
        self.mean_ap = average_precision_metric.MeanAveragePrecision(
            name='p3_AP')
        self.batch_size = 1
        self.shape = 20, 20, 3, 85

    def test_1_one_image_one_category(self):
        """1. 只有 1 张图片，只有 1 个类别，标签和预测结果完全相同，此时 AP 应该为 1。"""

        print('==' * 30)
        print('test_1_one_image_one_category')

        # 每个测试开始前，应该先把指标清零。
        self.mean_ap.reset_state()

        # label 形状为 (1, 19, 19, 3, 85)
        label = np.zeros(shape=(self.batch_size, *self.shape),
                         dtype=np.float32)

        # one_bbox 形状为 (85,)
        one_bbox = label[0, 10, 10, 0]

        # 设定 one_bbox 有物体。
        one_bbox[0] = 1
        # 设定 one_bbox 类别为 person。
        one_bbox[1] = 1
        # 设定 one_bbox 中心点为 10.2, 10.2，高度为 10，宽度为 10。
        one_bbox[-4:] = 10.2, 10.2, 10, 10

        iou = average_precision_metric.iou_calculator(label_bbox=label,
                                                      prediction_bbox=label)
        max_iou = np.amax(iou)
        print(f'The max_iou is: {max_iou}.')

        label = tf.convert_to_tensor(label)
        # 把数据输入给 update_state。
        self.mean_ap.update_state(y_true=label, y_pred=label,
                                  use_predictor=False)

        # 计算指标 mean_average_precision。
        mean_average_precision = self.mean_ap.result()
        print(f'The mean_average_precision is: {mean_average_precision}.')

        self.assertEqual(mean_average_precision, 1)

    def test_2_one_image_two_categories(self):
        """2. 只有 1 张图片，有 2 个类别，标签和预测结果完全相同，此时 AP 应该为 1。"""

        print('==' * 30)
        print('test_2_one_image_two_categories')

        # 每个测试开始前，应该先把指标清零。
        self.mean_ap.reset_state()

        # label 形状为 (1, 19, 19, 3, 85)
        label = np.zeros(shape=(self.batch_size, *self.shape),
                         dtype=np.float32)

        # bbox_one 形状为 (85,)
        bbox_one = label[0, 10, 10, 0]

        # 设定 bbox_one 有物体。
        bbox_one[0] = 1
        # 设定 bbox_one 类别为第 0 类， person。
        bbox_one[1] = 1
        # 设定 bbox_one 中心点为 10.2, 10.2，高度为 10，宽度为 10。
        bbox_one[-4:] = 10.2, 10.2, 10, 10

        # bbox_two 形状为 (85,)
        bbox_two = label[0, 10, 10, 1]

        # 设定 bbox_one 有物体。
        bbox_two[0] = 1
        # 设定 bbox_one 类别为第 1 类。
        bbox_two[2] = 1
        # 设定 bbox_one 中心点为 9.5, 9.5，高度为 5，宽度为 5。
        bbox_two[-4:] = 9.5, 9.5, 5, 5

        iou = average_precision_metric.iou_calculator(label_bbox=label,
                                                      prediction_bbox=label)
        max_iou = np.amax(iou)
        print(f'The max_iou is: {max_iou}.')

        label = tf.convert_to_tensor(label)
        # 把数据输入给 update_state。
        self.mean_ap.update_state(y_true=label, y_pred=label,
                                  use_predictor=False)

        # 计算指标 mean_average_precision。
        mean_average_precision = self.mean_ap.result()
        print(f'The mean_average_precision is: {mean_average_precision}.')

        self.assertEqual(mean_average_precision, 1)

    def test_3_one_image_low_iou(self):
        """3. 只有 1 张图片，只有 1 个类别，预测结果的 IoU 为 0.64，此时 AP 应该为 0.3。

        """

        print('==' * 30)
        print('test_3_one_image_low_iou')

        # 每个测试开始前，应该先把指标清零。
        self.mean_ap.reset_state()

        # label 形状为 (1, 19, 19, 3, 85)
        label = np.zeros(shape=(self.batch_size, *self.shape),
                         dtype=np.float32)

        # bbox_one 形状为 (85,)
        bbox_one = label[0, 10, 10, 0]

        # 设定 bbox_one 有物体。
        bbox_one[0] = 1
        # 设定 bbox_one 类别为第 0 类， person。
        bbox_one[1] = 1
        # 设定 bbox_one 中心点为 10.2, 10.2，高度为 10，宽度为 10。
        bbox_one[-4:] = 10.2, 10.2, 10, 10

        # prediction 形状为 (85,)
        prediction = label.copy()

        # 设定 prediction 的 bbox 中心点为 9.5, 9.5，高度为 8，宽度为 8。
        prediction[..., -4:] = 9.5, 9.5, 8, 8

        iou = average_precision_metric.iou_calculator(
            label_bbox=label, prediction_bbox=prediction)
        max_iou = np.amax(iou)
        print(f'The max_iou is: {max_iou}.')

        # 转换成张量，再用来计算。
        label = tf.convert_to_tensor(label)
        prediction = tf.convert_to_tensor(prediction)

        # 把数据输入给 update_state。
        self.mean_ap.update_state(y_true=label, y_pred=prediction,
                                  use_predictor=False)

        # 计算指标 mean_average_precision。
        mean_average_precision = self.mean_ap.result()
        print(f'The mean_average_precision is: {mean_average_precision}.')

        self.assertEqual(mean_average_precision, 0.3)

    def test_4_one_image_zero_ap(self):
        """4. 只有 1 张图片，只有 1 个类别，预测结果的 IoU 为 0.49，此时 AP 应该为 0。

        """

        print('==' * 30)
        print('test_4_one_image_zero_ap')

        # 每个测试开始前，应该先把指标清零。
        self.mean_ap.reset_state()

        # label 形状为 (1, 19, 19, 3, 85)
        label = np.zeros(shape=(self.batch_size, *self.shape),
                         dtype=np.float32)

        # bbox_one 形状为 (85,)
        bbox_one = label[0, 10, 10, 0]

        # 设定 bbox_one 有物体。
        bbox_one[0] = 1
        # 设定 bbox_one 类别为第 0 类， person。
        bbox_one[1] = 1
        # 设定 bbox_one 中心点为 10.2, 10.2，高度为 10，宽度为 10。
        bbox_one[-4:] = 10.2, 10.2, 10, 10

        # prediction 形状为 (85,)
        prediction = label.copy()

        # 设定 prediction 的 bbox 中心点为 9.5, 9.5，高度为 5，宽度为 5。
        prediction[..., -4:] = 9.5, 9.5, 7, 7

        iou = average_precision_metric.iou_calculator(
            label_bbox=label, prediction_bbox=prediction)
        max_iou = np.amax(iou)
        print(f'The max_iou is: {max_iou}.')

        # 转换成张量，再用来计算。
        label = tf.convert_to_tensor(label)
        prediction = tf.convert_to_tensor(prediction)

        # 把数据输入给 update_state。
        self.mean_ap.update_state(y_true=label, y_pred=prediction,
                                  use_predictor=False)

        # 计算指标 mean_average_precision。
        mean_average_precision = self.mean_ap.result()
        print(f'The mean_average_precision is: {mean_average_precision}.')

        self.assertEqual(mean_average_precision, 0)

    def test_5_1_one_image_low_objectness(self):
        """5.1 只有 1 张图片，只有 1 个类别，预测结果的 objectness 为 0.49，此时 AP
        应该为 0。
        """

        print('==' * 30)
        print('test_5_1_one_image_low_objectness')

        # 每个测试开始前，应该先把指标清零。
        self.mean_ap.reset_state()

        # label 形状为 (1, 19, 19, 3, 85)
        label = np.zeros(shape=(self.batch_size, *self.shape),
                         dtype=np.float32)

        # bbox_one 形状为 (85,)
        bbox_one = label[0, 10, 10, 0]

        # 设定 bbox_one 有物体。
        bbox_one[0] = 1
        # 设定 bbox_one 类别为第 0 类， person。
        bbox_one[1] = 1
        # 设定 bbox_one 中心点为 10.2, 10.2，高度为 10，宽度为 10。
        bbox_one[-4:] = 10.2, 10.2, 10, 10

        # prediction 形状为 (85,)
        prediction = label.copy()

        # prediction_bbox 形状为 (85,)。
        prediction_bbox = prediction[0, 10, 10, 0]
        # 设定 prediction 的 bbox 第 1 位为 0.49，即 objectness 为 0.49。
        prediction_bbox[0] = 0.49

        iou = average_precision_metric.iou_calculator(
            label_bbox=label, prediction_bbox=prediction)
        max_iou = np.amax(iou)
        print(f'The max_iou is: {max_iou}.')

        # 转换成张量，再用来计算。
        label = tf.convert_to_tensor(label)
        prediction = tf.convert_to_tensor(prediction)

        # 把数据输入给 update_state。
        self.mean_ap.update_state(y_true=label, y_pred=prediction,
                                  use_predictor=False)

        # 计算指标 mean_average_precision。
        mean_average_precision = self.mean_ap.result()
        print(f'The mean_average_precision is: {mean_average_precision}.')

        self.assertEqual(mean_average_precision, 0)

    def test_5_2_one_image_two_predictions_one_low_objectness(self):
        """5.2 只有 1 张图片，只有 1 个类别，但是有 2 个预测结果。第一个正确预测结果 bbox
        和标签完全相同，第二个错误预测结果（因为标签中没有这个物体，所以是预测结果错误）
        bbox，objectness 和 classification 都为 0.51，此时 AP 应该为 0.75。
        """

        print('==' * 30)
        print('test_5_2_one_image_two_predictions_one_low_objectness')

        # 每个测试开始前，应该先把指标清零。
        self.mean_ap.reset_state()

        # label 形状为 (1, 19, 19, 3, 85)。
        label = np.zeros(shape=(self.batch_size, *self.shape),
                         dtype=np.float32)

        # bbox_one 形状为 (85,)
        bbox_one = label[0, 10, 10, 0]

        # 设定 bbox_one 有物体。
        bbox_one[0] = 1
        # 设定 bbox_one 类别为第 0 类， person。
        bbox_one[1] = 1
        # 设定 bbox_one 中心点为 10.2, 10.2，高度为 10，宽度为 10。
        bbox_one[-4:] = 10.2, 10.2, 10, 10

        # prediction 形状为 (1, 19, 19, 3, 85)。
        prediction = label.copy()

        # prediction 有 2 个预测结果，将错误的预测命名为 prediction_bbox_wrong。
        # prediction_bbox_wrong 形状为 (85,)。
        prediction_bbox_wrong = prediction[0, 10, 10, 1]

        # 设定 prediction_bbox_wrong 的第 0 位为 0.51，即 objectness 为 0.51。
        prediction_bbox_wrong[0] = 0.51
        # 设定 prediction_bbox_wrong 的第 1 位为 0.51，即类别为第 0 类，置信度为 0.51。
        prediction_bbox_wrong[1] = 0.51

        # 设定 prediction_bbox_wrong 中心点为 10.2, 10.2，高度为 9.9，宽度为 9.9。
        # 注意这里高度宽度不能设置和预测结果的另一个 bbox 完全相同，因为这会导致最大 IoU
        # 的布尔张量有 2 个 True 位置，也就是 max_iou_position 有 2 个 True 值，
        # 从而一次去掉了预测结果中的 2 个 bboxes，使得计算不正确。当然在实际情况中不会
        # 出现这个问题，因为预测结果是浮点数，两个预测结果并不会完全相同。
        prediction_bbox_wrong[-4:] = 10.2, 10.2, 9.9, 9.9

        iou = average_precision_metric.iou_calculator(
            label_bbox=label, prediction_bbox=prediction)
        max_iou = np.amax(iou)
        print(f'The max_iou is: {max_iou}.')

        # 转换成张量，再用来计算。
        label = tf.convert_to_tensor(label)
        prediction = tf.convert_to_tensor(prediction)

        # 把数据输入给 update_state。
        self.mean_ap.update_state(y_true=label, y_pred=prediction,
                                  use_predictor=False)

        # 计算指标 mean_average_precision。
        mean_average_precision = self.mean_ap.result()
        print(f'The mean_average_precision is: {mean_average_precision}.')

        self.assertEqual(mean_average_precision, 0.75)

    def test_6_one_image_low_classification_confidence(self):
        """6. 只有 1 张图片，只有 1 个类别，预测结果的类别置信度为 0.49，此时 AP 应该
        为 0。
        """

        print('==' * 30)
        print('test_6_one_image_low_classification_confidence')

        # 每个测试开始前，应该先把指标清零。
        self.mean_ap.reset_state()

        # label 形状为 (1, 19, 19, 3, 85)
        label = np.zeros(shape=(self.batch_size, *self.shape),
                         dtype=np.float32)

        # bbox_one 形状为 (85,)
        bbox_one = label[0, 10, 10, 0]

        # 设定 bbox_one 有物体。
        bbox_one[0] = 1
        # 设定 bbox_one 类别为第 0 类， person。
        bbox_one[1] = 1
        # 设定 bbox_one 中心点为 10.2, 10.2，高度为 10，宽度为 10。
        bbox_one[-4:] = 10.2, 10.2, 10, 10

        # prediction 形状为 (85,)
        prediction = label.copy()

        # prediction_bbox 形状为 (85,)。
        prediction_bbox = prediction[0, 10, 10, 0]
        # 设定 prediction 的 bbox 第 1 位为 0.49，即类别置信度为 0.49。
        prediction_bbox[1] = 0.49

        iou = average_precision_metric.iou_calculator(
            label_bbox=label, prediction_bbox=prediction)
        max_iou = np.amax(iou)
        print(f'The max_iou is: {max_iou}.')

        # 转换成张量，再用来计算。
        label = tf.convert_to_tensor(label)
        prediction = tf.convert_to_tensor(prediction)

        # 把数据输入给 update_state。
        self.mean_ap.update_state(y_true=label, y_pred=prediction,
                                  use_predictor=False)

        # 计算指标 mean_average_precision。
        mean_average_precision = self.mean_ap.result()
        print(f'The mean_average_precision is: {mean_average_precision}.')

        self.assertEqual(mean_average_precision, 0)

    def test_7_two_images_one_category(self):
        """7. 有 2 张图片，只有 1 个类别，标签和预测结果完全相同，此时 AP 应该为 1。"""

        print('==' * 30)
        print('test_7_two_images_one_category')

        # 每个测试开始前，应该先把指标清零。
        self.mean_ap.reset_state()

        # label 形状为 (2, 19, 19, 3, 85)
        label = np.zeros(shape=(2, *self.shape),
                         dtype=np.float32)

        # one_image_bbox 形状为 (85,)
        one_image_bbox = label[0, 10, 10, 0]

        # 设定 one_image_bbox 有物体。
        one_image_bbox[0] = 1
        # 设定 one_image_bbox 类别为 person。
        one_image_bbox[1] = 1
        # 设定 one_image_bbox 中心点为 10.2, 10.2，高度为 10，宽度为 10。
        one_image_bbox[-4:] = 10.2, 10.2, 10, 10

        # 下面设定第 2 张图片的 bbox，使其和第一张图片的 bbox 完全相同即可。
        label[1, 10, 10, 0] = one_image_bbox

        iou = average_precision_metric.iou_calculator(
            label_bbox=label, prediction_bbox=label)
        max_iou = np.amax(iou)
        print(f'The max_iou is: {max_iou}.')

        label = tf.convert_to_tensor(label)
        # 把数据输入给 update_state。
        self.mean_ap.update_state(y_true=label, y_pred=label,
                                  use_predictor=False)

        # 计算指标 mean_average_precision。
        mean_average_precision = self.mean_ap.result()
        print(f'The mean_average_precision is: {mean_average_precision}.')

        self.assertEqual(mean_average_precision, 1)

    def test_8_two_images_one_zero_ap(self):
        """8. 有 2 张图片，只有 1 个类别，一张图片预测结果的 IoU 为 0.49，则其 AP 将
        为 0，另一张图片 IoU 为 1，此时最终 AP 应为 0.375。"""

        print('==' * 30)
        print('test_8_two_images_one_zero_ap')

        # 每个测试开始前，应该先把指标清零。
        self.mean_ap.reset_state()

        # label 形状为 (2, 19, 19, 3, 85)
        label = np.zeros(shape=(2, *self.shape),
                         dtype=np.float32)

        # one_image_bbox 形状为 (85,)
        one_image_bbox = label[0, 10, 10, 0]

        # 设定 one_image_bbox 有物体。
        one_image_bbox[0] = 1
        # 设定 one_image_bbox 类别为 person。
        one_image_bbox[1] = 1
        # 设定 one_image_bbox 中心点为 10.2, 10.2，高度为 10，宽度为 10。
        one_image_bbox[-4:] = 10.2, 10.2, 10, 10

        # 下面设定第 2 张图片的 bbox，使其和第一张图片的 bbox 完全相同即可。
        label[1, 10, 10, 0] = one_image_bbox

        # prediction 形状为 (2, 19, 19, 3, 85)
        prediction = label.copy()

        # 设定第 2 张图片的 bbox 类别置信度为 0.99。
        prediction[1, 10, 10, 0, 1] = 0.99
        # 设定第 2 张图片的 bbox 中心点为 9.5, 9.5，高度为 7，宽度为 7，则 IoU 为 0.49。
        prediction[1, 10, 10, 0, -4:] = 9.5, 9.5, 7, 7

        iou = average_precision_metric.iou_calculator(
            label_bbox=label, prediction_bbox=prediction)
        max_iou = np.amax(iou)
        print(f'The max_iou is: {max_iou}.')

        label = tf.convert_to_tensor(label)
        prediction = tf.convert_to_tensor(prediction)

        # 把数据输入给 update_state。
        self.mean_ap.update_state(y_true=label, y_pred=prediction,
                                  use_predictor=False)

        # 计算指标 mean_average_precision。
        mean_average_precision = self.mean_ap.result()
        print(f'The mean_average_precision is: {mean_average_precision}.')

        self.assertEqual(mean_average_precision, 0.375)

    def test_9_one_objectness_below_threshold(self):
        """9. 有 2 张图片，只有 1 个类别。预测结果的一个 bbox 和标签完全相同，另一个
         objectness 为 0.49，此时最终 AP 应为 0.5。"""

        print('==' * 30)
        print('test_9_one_objectness_below_threshold')

        # 每个测试开始前，应该先把指标清零。
        self.mean_ap.reset_state()

        # label 形状为 (2, 19, 19, 3, 85)
        label = np.zeros(shape=(2, *self.shape),
                         dtype=np.float32)

        # one_image_bbox 形状为 (85,)
        one_image_bbox = label[0, 10, 10, 0]

        # 设定 one_image_bbox 有物体。
        one_image_bbox[0] = 1
        # 设定 one_image_bbox 类别为 person。
        one_image_bbox[1] = 1
        # 设定 one_image_bbox 中心点为 10.2, 10.2，高度为 10，宽度为 10。
        one_image_bbox[-4:] = 10.2, 10.2, 10, 10

        # 下面设定第 2 张图片的 bbox，使其和第一张图片的 bbox 完全相同即可。
        label[1, 10, 10, 0] = one_image_bbox

        # prediction 形状为 (2, 19, 19, 3, 85)
        prediction = label.copy()

        # 设定第 2 张图片的 bbox 类别置信度为 0.49。
        prediction[1, 10, 10, 0, 1] = 0.49

        iou = average_precision_metric.iou_calculator(
            label_bbox=label, prediction_bbox=prediction)
        max_iou = np.amax(iou)
        print(f'The max_iou is: {max_iou}.')

        label = tf.convert_to_tensor(label)
        prediction = tf.convert_to_tensor(prediction)

        # 把数据输入给 update_state。
        self.mean_ap.update_state(y_true=label, y_pred=prediction,
                                  use_predictor=False)

        # 计算指标 mean_average_precision。
        mean_average_precision = self.mean_ap.result()
        print(f'The mean_average_precision is: {mean_average_precision}.')

        self.assertEqual(mean_average_precision, 0.5)

    def test_10_classification_confidence_below_threshold(self):
        """10. 有 2 张图片，只有 1 个类别，一张图片的类别置信度为 0.49，则其 AP 将
        为 0，此时最终 AP 应为 0.5。"""

        print('==' * 30)
        print('test_10_classification_confidence_below_threshold')

        # 每个测试开始前，应该先把指标清零。
        self.mean_ap.reset_state()

        # label 形状为 (2, 19, 19, 3, 85)
        label = np.zeros(shape=(2, *self.shape),
                         dtype=np.float32)

        # one_image_bbox 形状为 (85,)
        one_image_bbox = label[0, 10, 10, 0]

        # 设定 one_image_bbox 有物体。
        one_image_bbox[0] = 1
        # 设定 one_image_bbox 类别为 person。
        one_image_bbox[1] = 1
        # 设定 one_image_bbox 中心点为 10.2, 10.2，高度为 10，宽度为 10。
        one_image_bbox[-4:] = 10.2, 10.2, 10, 10

        # 下面设定第 2 张图片的 bbox，使其和第一张图片的 bbox 完全相同即可。
        label[1, 10, 10, 0] = one_image_bbox

        # prediction 形状为 (2, 19, 19, 3, 85)
        prediction = label.copy()

        # 设定第 2 张图片的 bbox 类别置信度为 0.49。
        prediction[1, 10, 10, 0, 1] = 0.49

        iou = average_precision_metric.iou_calculator(
            label_bbox=label, prediction_bbox=prediction)
        max_iou = np.amax(iou)
        print(f'The max_iou is: {max_iou}.')

        label = tf.convert_to_tensor(label)
        prediction = tf.convert_to_tensor(prediction)

        # 把数据输入给 update_state。
        self.mean_ap.update_state(y_true=label, y_pred=prediction,
                                  use_predictor=False)

        # 计算指标 mean_average_precision。
        mean_average_precision = self.mean_ap.result()
        print(f'The mean_average_precision is: {mean_average_precision}.')

        self.assertEqual(mean_average_precision, 0.5)

    def test_11_two_categories_two_images(self):
        """11. 有 2 个类别，每个类别都有 2 张图片。一个类别的 AP 为 0.375，另一个类别
        的 AP 为 1，最终 AP 应该为 0.6875。"""

        print('==' * 30)
        print('test_11_two_categories_two_images')

        # 每个测试开始前，应该先把指标清零。
        self.mean_ap.reset_state()

        # label 形状为 (2, 19, 19, 3, 85)
        label = np.zeros(shape=(2, *self.shape),
                         dtype=np.float32)

        # 设定第一张图片中的第 1 个类别。
        # image_one_category_one 形状为 (85,).
        image_one_category_one = label[0, 10, 10, 0]

        # 设定 image_one_category_one 有物体。
        image_one_category_one[0] = 1
        # 设定 image_one_category_one 类别为 person。
        image_one_category_one[1] = 1
        # 设定 image_one_category_one 中心点为 10.2, 10.2，高度为 10，宽度为 10。
        image_one_category_one[-4:] = 10.2, 10.2, 10, 10

        # 设定第一张图片中的第 2 个类别。
        label[0, 10, 10, 1] = image_one_category_one.copy()

        # image_one_category_two 形状为 (85,).
        image_one_category_two = label[0, 10, 10, 1]

        # 修改 image_one_category_two 类别，从第 0 类改为第 1 类。
        image_one_category_two[1] = 0
        image_one_category_two[2] = 1

        # 下面设定第 2 张图片的标签，使其和第一张图片完全相同即可。
        label[1] = label[0].copy()

        # prediction 形状为 (2, 19, 19, 3, 85)
        prediction = label.copy()

        # 设定第 1 个类别的第 1 张图片的 bbox 类别置信度为 0.99。
        prediction[0, 10, 10, 0, 1] = 0.99
        # 设定第 1 个类别的第 1 张图片的 bbox 中心点为 9.5, 9.5，高度为 7，宽度为 7，
        # 则 IoU 为 0.49，其 AP 为 0.375. 第二个类别 IoU 为 1，其 AP 也为 1.
        prediction[0, 10, 10, 0, -4:] = 9.5, 9.5, 7, 7

        iou = average_precision_metric.iou_calculator(
            label_bbox=label, prediction_bbox=prediction)
        max_iou = np.amax(iou)
        print(f'The max_iou is: {max_iou}.')

        label = tf.convert_to_tensor(label)
        prediction = tf.convert_to_tensor(prediction)

        # 把数据输入给 update_state。
        self.mean_ap.update_state(y_true=label, y_pred=prediction,
                                  use_predictor=False)

        # 计算指标 mean_average_precision。
        mean_average_precision = self.mean_ap.result()
        print(f'The mean_average_precision is: {mean_average_precision}.')

        self.assertEqual(mean_average_precision, 0.6875)

    # 在 average_precision_metric 运行时，不要运行下面的 reset_metric 测试，以免状态量
    # 被清零，所以这里设置默认跳过该测试。确定需要测试该项时，将下一行 @unittest.skip
    # 注释掉即可。
    @unittest.skip
    def test_12_reset_metric(self):
        """12. 测试指标清零后的状态。注意该测试会把 average_precision_metric 中的 3 个状态量清零。"""

        print('==' * 30)
        print('test_12_reset_metric')
        # 先将指标清零。
        self.mean_ap.reset_state()

        self.assertTrue(tf.math.reduce_all(tf.experimental.numpy.isclose(
            average_precision_metric.latest_positive_bboxes, 0)))

        self.assertTrue(tf.math.reduce_all(tf.experimental.numpy.isclose(
            average_precision_metric.labels_quantity_per_image, 0)))

        self.assertFalse(
            tf.math.reduce_all(average_precision_metric.showed_up_classes))


if __name__ == '__main__':
    unittest.main()

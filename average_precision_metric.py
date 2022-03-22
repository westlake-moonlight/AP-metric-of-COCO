"""Build the AP metric of COCO dataset in Keras/TensorFlow 2.8."""

import numpy as np
import tensorflow as tf

# 设置如下全局变量，用大写字母。

CLASSES = 80  # 如果使用 COCO 数据集，则需要探测 80 个类别。

# 为了获得更快的速度，使用小的特征图。原始模型的 p5 特征图为 19x19，模型输入图片为 608x608
FEATURE_MAP_P5 = np.array((10, 10))  # 19, 19
FEATURE_MAP_P4 = FEATURE_MAP_P5 * 2  # 38, 38
FEATURE_MAP_P3 = FEATURE_MAP_P4 * 2  # 76, 76

# 格式为 height, width。当两者大小不同时尤其要注意。是 FEATURE_MAP_P3 的 8 倍。
MODEL_IMAGE_SIZE = FEATURE_MAP_P3 * 8  # 608, 608

# 如果使用不同大小的 FEATURE_MAP，应该相应调整预设框的大小。
resize_scale = 19 / FEATURE_MAP_P5[0]

# 根据 YOLO V3 论文的 2.3 节，设置 ANCHOR_BOXES 。除以比例后取整数部分。
ANCHOR_BOXES_P5 = [(116 // resize_scale, 90 // resize_scale),
                   (156 // resize_scale, 198 // resize_scale),
                   (373 // resize_scale, 326 // resize_scale)]
ANCHOR_BOXES_P4 = [(30 // resize_scale, 61 // resize_scale),
                   (62 // resize_scale, 45 // resize_scale),
                   (59 // resize_scale, 119 // resize_scale)]
ANCHOR_BOXES_P3 = [(10 // resize_scale, 13 // resize_scale),
                   (16 // resize_scale, 30 // resize_scale),
                   (33 // resize_scale, 23 // resize_scale)]

EPSILON = 1e-10


def _transform_predictions(prediction, anchor_boxes):
    """将模型的预测结果转换为模型输入的图片大小，可以在全局变量 MODEL_IMAGE_SIZE 中
    设置模型输入图片大小。

    将每个长度为 85 的预测结果进行转换，第 0 位为置信度 confidence，第 1 位到第 81
    位为分类的 one-hot 编码。置信度和分类结果都需要用 sigmoid 转换为 [0, 1]之间的数，
    相当于转换为概率值。最后 4 位是探测框的预测结果，需要根据 YOLO V3 论文进行转换。
    倒数第 4 位到倒数第 3 位为预测框的中心点，需要对中心点用 sigmoid 函数进行转换。然
    后乘以一个比例，就得到中心点在模型输入图片中的实际坐标值。
    倒数第 2 位和最后一位是探测框的宽度和高度，需要先用指数函数转换为非负数，再乘以探测
    框的高度和宽度。

    Arguments:
        prediction: 一个 3D 张量，形状为 (N, *batch_size_feature_map, 255)，是
            模型的 3 个预测结果张量之一。height, width 是特征图大小。使用时 Keras
            将会自动插入一个第 0 维度，作为批量维度。
        anchor_boxes: 一个元祖，其中包含 3 个元祖，代表了当前 prediction 所对应
            的 3 个预设框。

    Returns:
        transformed_prediction: 一个 4D 张量，形状为 (*FEATURE_MAP_Px, 3, 85)。
            FEATURE_MAP_Px 是 p5, p4, p3 特征图大小。3 是特征图每个位置上，预设框
            的数量。85 是单个预测结果的长度。
            长度为 85 的预测向量，第 0 为表示是否有物体的概率， 第 1 位到第 80 位，
            是表示物体类别的 one-hot 编码，而最后 4 位，则分别是物体框 bbox 的参数
            (center_x, center_y, height, width)。
    """

    # 下面的注释以 p5 为例。在 p5 的 19x19 个位置上，将每个长度为 255 的向量，转换为
    #  3x85 的形状。
    #  85 位分别表示 [confidence, classification..., tx, ty, th, tw]，其中
    #  classification 部分，共包含 80 位，最后 4 位是探测框的中心点坐标和大小。

    # prediction 的形状为 (N, *batch_size_feature_map, 255), N 为 batch_size。
    # 进行 reshape时，必须带上 batch_size，所以用 batch_size_feature_map
    batch_size_feature_map = prediction.shape[: 3]
    prediction = tf.reshape(prediction,
                            shape=(*batch_size_feature_map, 3, 85))

    # get_probability 形状为 (N, 19, 19, 3, 81)，包括置信度和分类结果两部分。
    get_probability = tf.math.sigmoid(prediction[..., : 81])

    # confidence 形状为 (N, 19, 19, 3, 1) 需要配合使用 from_logits=False
    confidence = get_probability[..., : 1]

    # classification 形状为 (N, 19, 19, 3, 80)，需要配合使用 from_logits=False
    classification = get_probability[..., 1: 81]

    # prediction 的形状为 (N, 19, 19, 3, 85), N 为 batch_size。
    feature_map = prediction.shape[1: 3]

    # 根据 YOLO V3 论文中的 figure 2，需要对 bbox 坐标和尺寸进行转换。tx_ty 等标
    # 注记号和论文的记号一一对应。
    # tx_ty 形状为 (N, 19, 19, 3, 2)，分别代表 tx, ty。
    tx_ty = prediction[..., -4: -2]
    # 根据 YOLO V3论文，需要先取得 cx_cy。cx_cy 实际是一个比例值，在计算 IOU 和损失
    # 值之前，应该转换为 608x608 大小图片中的实际值。
    # 注意，根据论文 2.1 节第一段以及配图 figure 2，cx_cy 其实是每一个 cell
    # 的左上角点，这样预测框的中心点 bx_by 才能达到该 cell 中的每一个位置。
    grid = tf.ones(shape=feature_map)  # 构造一个 19x19 的网格
    cx_cy = tf.where(grid)  # where 函数可以获取张量的索引值，也就是 cx, cy

    # 为了使用混合精度计算，该函数内部定义的数据类型，必须和 prediction 的数据类型一致。
    compute_dtype = prediction.dtype

    cx_cy = tf.cast(x=cx_cy, dtype=compute_dtype)  # cx_cy 原本是 int64 类型

    # cx_cy 的形状为 (361, 2)， 361 = 19 x 19，下面将其形状变为 (1, 19, 19, 1, 2)
    cx_cy = tf.reshape(cx_cy, shape=(1, *feature_map, 2))
    cx_cy = cx_cy[..., tf.newaxis, :]  # 展示一下 tf.newaxis 的用法

    #  cx_cy 的形状为 (1, 19, 19, 1, 2), tx_ty 的形状为 (N, 19, 19, 3, 2)
    bx_by = tf.math.sigmoid(tx_ty) + cx_cy

    # 下面根据 th, tw, 计算 bh, bw。th_tw 形状为 (N, 19, 19, 3, 2)
    th_tw = prediction[..., -2:]
    ph_pw = tf.convert_to_tensor(anchor_boxes, dtype=compute_dtype)
    # ph_pw 的形状为 (3, 2)，和上面的 cx_cy 同理，需要将 ph_pw 的形状变为
    # (1, 1, 3, 2)
    ph_pw = tf.reshape(ph_pw, shape=(1, 1, 1, 3, 2))
    # 此时 ph_pw 和 th_tw 的张量阶数 rank 相同，会自动扩展 broadcast，进行算术运算。
    bh_bw = ph_pw * tf.math.exp(th_tw)

    # 在计算 CIOU 损失时，如果高度宽度过大，计算预测框面积会产生 NaN 值，导致模型无法
    # 训练。所以把预测框的高度宽度限制到不超过图片大小即可。
    bh_bw = tf.clip_by_value(
        bh_bw, clip_value_min=0, clip_value_max=MODEL_IMAGE_SIZE[0])

    # bx_by，bh_bw 为比例值，需要转换为在 608x608 大小图片中的实际值。
    image_scale_height = MODEL_IMAGE_SIZE[0] / feature_map[0]
    image_scale_width = MODEL_IMAGE_SIZE[1] / feature_map[1]
    image_scale = image_scale_height, image_scale_width

    # bx_by 是一个比例值，乘以比例 image_scale 之后，bx_by 将代表图片中实际
    # 的长度数值。比如此时 bx, by 的数值可能是 520， 600 等，数值范围 [0, 608]
    # 而 bh_bw 已经是一个长度值，不需要再乘以比例。
    bx_by *= image_scale

    bx_by = tf.clip_by_value(
        bx_by, clip_value_min=0, clip_value_max=MODEL_IMAGE_SIZE[0])

    transformed_prediction = tf.concat(
        values=[confidence, classification, bx_by, bh_bw], axis=-1)

    return transformed_prediction


def predictor(inputs):
    """对模型输出的 1 个 head 进行转换。

    转换方式为：
    先将 head 的形状从 (batch_size, height, width, 255) 变为 (batch_size, height,
    width, 3, 85)。将每个长度为 85 的预测结果进行转换，第 0 位为置信度 confidence，
    第 1 位到第 81位为分类的 one-hot 编码，均需要用 sigmoid 转换为 [0, 1] 之间的数。
    最后 4 位是探测框的预测结果，需要根据 YOLO V3 论文进行转换。
    倒数第 4 位到倒数第 3 位为预测框的中心点，需要对中心点用 sigmoid 函数进行转换。然
    后乘以一个比例，就得到中心点在模型输入图片中的实际坐标值。
    倒数第 2 位和最后一位是探测框的宽度和高度，需要先用指数函数转换为非负数，再乘以探测
    框的高度和宽度。

    Arguments:
        inputs: 一个元祖，包含来自 Heads 输出的 3个 3D 张量。分别表示为
            p5_head, p4_head, p3_head。使用时 Keras 将会自动插入一个第 0 维度，
            作为批量维度。
    Returns:
        p5_prediction: 一个 5D 张量，形状为 (batch_size, height, width, 3, 85)。
            height, width 是特征图大小。3 是特征图的每个位置上，预设框的数量。
            85 是单个预测结果的长度。下面 p4_prediction， p3_prediction 也是一样。
            p5_prediction 的 height, width 为 19, 19.
        p4_prediction: 一个 5D 张量，形状为 (batch_size, 38, 38, 3, 85)。
        p3_prediction: 一个 5D 张量，形状为 (batch_size, 76, 76, 3, 85)。
    """

    # px_head 代表 p5_head, p4_head, p3_head
    px_head = inputs
    feature_map_size = px_head.shape[1: 3]

    anchor_boxes_px = None
    if feature_map_size == (*FEATURE_MAP_P5,):
        anchor_boxes_px = ANCHOR_BOXES_P5
    elif feature_map_size == (*FEATURE_MAP_P4,):
        anchor_boxes_px = ANCHOR_BOXES_P4
    elif feature_map_size == (*FEATURE_MAP_P3,):
        anchor_boxes_px = ANCHOR_BOXES_P3

    px_prediction = _transform_predictions(px_head, anchor_boxes_px)

    return px_prediction


def iou_calculator(label_bbox, prediction_bbox):
    """计算预测框和真实框的 IoU 。

    用法说明：使用时，要求输入的 label_bbox, prediction_bbox 形状相同，均为 4D 张量。将
    在两个输入的一一对应的位置上，计算 IoU。
    举例来说，假如两个输入的形状都是 (19, 19, 3, 4)，而标签 label_bbox 只在 (8, 12, 0)
    位置有一个物体框，则 iou_calculator 将会寻找 prediction_bbox 在同样位置
    (8, 12, 0) 的物体框，并计算这两个物体框之间的 IoU。prediction_bbox 中其它位置的物
    体框，并不会和 label_bbox 中 (8, 12, 0) 位置的物体框计算 IoU。
    计算结果的形状为 (19, 19, 3)，并且将在 (8, 12, 0) 位置有一个 IoU 值。

    Arguments:
        label_bbox: 一个 4D 张量，形状为 (input_height, input_width, 3, 4)，代表标
            签中的物体框。
            最后一个维度的 4 个值分别代表物体框的 (center_x, center_y, height_bbox,
            width_bbox)。第 2 个维度的 3 表示有 3 种不同宽高比的物体框。
            该 4 个值必须是实际值，而不是比例值。
        prediction_bbox: 一个 4D 张量，形状为 (input_height, input_width, 3, 4)，
            代表预测结果中的物体框。最后一个维度的 4 个值分别代表物体框的
            (center_x, center_y, height_bbox, width_bbox)。第 2 个维度的 3 表示
            有 3 种不同宽高比的物体框。该 4 个值必须是实际值，而不是比例值。
    Returns:
        iou: 一个 3D 张量，形状为 (input_height, input_width, 3)，代表交并比 IoU。
    """

    # 两个矩形框 a 和 b 相交时，要同时满足的 4 个条件是：
    # left_edge_a < right_edge_b , right_edge_a > left_edge_b
    # top_edge_a < bottom_edge_b , bottom_edge_a > top_edge_b

    # 对每个 bbox，先求出 4 条边。left_edge，right_edge 形状为
    # (input_height, input_width, 3)
    label_left_edge = label_bbox[..., -4] - label_bbox[..., -1] / 2
    label_right_edge = label_bbox[..., -4] + label_bbox[..., -1] / 2

    prediction_left_edge = (prediction_bbox[..., -4] -
                            prediction_bbox[..., -1] / 2)
    prediction_right_edge = (prediction_bbox[..., -4] +
                             prediction_bbox[..., -1] / 2)

    label_top_edge = label_bbox[..., -3] - label_bbox[..., -2] / 2
    label_bottom_edge = label_bbox[..., -3] + label_bbox[..., -2] / 2

    prediction_top_edge = (prediction_bbox[..., -3] -
                           prediction_bbox[..., -2] / 2)
    prediction_bottom_edge = (prediction_bbox[..., -3] +
                              prediction_bbox[..., -2] / 2)

    # left_right_condition 的形状为 (input_height, input_width, 3)
    # 表示 2 个条件：left_edge_a < right_edge_b , right_edge_a > left_edge_b
    left_right_condition = tf.math.logical_and(
        x=(label_left_edge < prediction_right_edge),
        y=(label_right_edge > prediction_left_edge))
    # top_bottom_condition 的形状为 (input_height, input_width, 3)
    # 表示 2 个条件：top_edge_a < bottom_edge_b , bottom_edge_a > top_edge_b
    top_bottom_condition = tf.math.logical_and(
        x=(label_top_edge < prediction_bottom_edge),
        y=(label_bottom_edge > prediction_top_edge))

    # intersection_condition 的形状为
    # (input_height, input_width, 3)，是 4 个条件的总和
    intersection_condition = tf.math.logical_and(x=left_right_condition,
                                                 y=top_bottom_condition)
    # 形状扩展为 (input_height, input_width, 3, 1)
    intersection_condition = tf.expand_dims(intersection_condition, axis=-1)
    # 形状扩展为 (input_height, input_width, 3, 4)
    intersection_condition = tf.repeat(input=intersection_condition,
                                       repeats=4, axis=-1)

    # horizontal_edges, vertical_edges 的形状为
    # (input_height, input_width, 3, 4)
    horizontal_edges = tf.stack(
        values=[label_top_edge, label_bottom_edge,
                prediction_top_edge, prediction_bottom_edge], axis=-1)

    vertical_edges = tf.stack(
        values=[label_left_edge, label_right_edge,
                prediction_left_edge, prediction_right_edge], axis=-1)

    zero_pad_edges = tf.zeros_like(input=horizontal_edges)
    # 下面使用 tf.where，可以使得 horizontal_edges 和 vertical_edges 的形状保持为
    # (input_height, input_width, 3, 4)，并且只保留相交 bbox 的边长值，其它设为 0
    horizontal_edges = tf.where(condition=intersection_condition,
                                x=horizontal_edges, y=zero_pad_edges)
    vertical_edges = tf.where(condition=intersection_condition,
                              x=vertical_edges, y=zero_pad_edges)

    horizontal_edges = tf.sort(values=horizontal_edges, axis=-1)
    vertical_edges = tf.sort(values=vertical_edges, axis=-1)

    # 4 条边按照从小到大的顺序排列后，就可以把第二大的减去第三大的边，得到边长。
    # intersection_height, intersection_width 的形状为
    # (input_height, input_width, 3)
    intersection_height = horizontal_edges[..., -2] - horizontal_edges[..., -3]
    intersection_width = vertical_edges[..., -2] - vertical_edges[..., -3]

    # intersection_area 的形状为 (input_height, input_width, 3)
    intersection_area = intersection_height * intersection_width

    prediction_bbox_width = prediction_bbox[..., -1]
    prediction_bbox_height = prediction_bbox[..., -2]

    # 不能使用混合精度计算。因为 float16 格式下，数值达到 65520 时，就会溢出变为 inf，
    # 从而导致 NaN。而 prediction_bbox_area 的数值是可能达到 320*320 甚至更大的。
    prediction_bbox_area = prediction_bbox_width * prediction_bbox_height

    label_bbox_area = label_bbox[..., -1] * label_bbox[..., -2]

    # union_area 的形状为 (input_height, input_width, 3)
    union_area = prediction_bbox_area + label_bbox_area - intersection_area

    # 为了计算的稳定性，避免出现 nan、inf 的情况，分母可能为 0 时应加上一个极小量 EPSILON
    # iou 的形状为 (input_height, input_width, 3)
    iou = intersection_area / (union_area + EPSILON)

    return iou


# 指标 MeanAveragePrecision 用到的全局变量，使用大写字母。
# OBJECTNESS_THRESHOLD: 一个浮点数，表示物体框内，是否存在物体的置信度阈值。
OBJECTNESS_THRESHOLD = 0.5
# CLASSIFICATION_CONFIDENCE_THRESHOLD: 一个浮点数，表示物体框的类别置信度阈值。
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.5


# LATEST_RELATED_IMAGES: 一个整数，表示最多使用多少张相关图片来计算一个类别的 AP。
LATEST_RELATED_IMAGES = 3
# BBOXES_PER_IMAGE: 一个整数，表示对于一个类别的每张相关图片，最多使用
# BBOXES_PER_IMAGE 个 bboxes 来计算 AP。
BBOXES_PER_IMAGE = 20

# latest_positive_bboxes: 一个 tf.Variable 张量，用于存放最近的
# LATEST_RELATED_IMAGES 张相关图片，且每张图片只保留 BBOXES_PER_IMAGE 个
# positive bboxes，每个 bboxes 有 2 个数值，分别是类别置信度，以及 IoU 值。
latest_positive_bboxes = tf.Variable(
    tf.zeros(shape=(CLASSES, LATEST_RELATED_IMAGES, BBOXES_PER_IMAGE, 2)),
    trainable=False, name='latest_positive_bboxes')

# labels_quantity_per_image: 一个形状为 (CLASSES, BBOXES_PER_IMAGE) 的整数型
# 张量，表示每张图片中，该类别的标签 bboxes 数量。
labels_quantity_per_image = tf.Variable(
    tf.zeros(shape=(CLASSES, LATEST_RELATED_IMAGES)),
    trainable=False, name='labels_quantity_per_image')

# showed_up_classes：一个形状为 (CLASSES, ) 的布尔张量，用于记录所有出现过的类别。
# 每批次数据中，都会出现不同的类别，计算指标时，只使用出现过的类别进行计算。
showed_up_classes = tf.Variable(tf.zeros(shape=(CLASSES,), dtype=tf.bool),
                                trainable=False, name='showed_up_classes')


class MeanAveragePrecision(tf.keras.metrics.Metric):
    """计算 COCO 的 AP 指标。

    使用说明：COCO 的 AP 指标，是 10 个 IoU 阈值下，80 个类别 AP 的平均值，即 mean
    average precision。为了和单个类别的 AP 进行区分，这里使用 mAP 来代表 AP 的平均值。

    受内存大小的限制，对每一个类别，只使用最近 LATEST_RELATED_IMAGES 张相关图片计算其
    AP(COCO 实际是使用所有相关图片)。
    相关图片是指该图片的标签或是预测结果的正样本中，包含了该类别。对每个类别的每张图片，
    只保留 BBOXES_PER_IMAGE 个 bboxes 来计算 AP（COCO 实际是最多使用 100 个 bboxes）。
    """

    def __init__(self, name='AP', **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None,
                     use_predictor=True):
        """根据每个 batch 的计算结果，区分 4 种情况，更新状态 state。

        Arguments:
            y_true: 一个浮点类型张量，形状为 (batch_size, *Feature_Map_px, 3, 85)。
                是每个批次数据的标签。
            y_pred: 一个浮点类型张量，形状为 (batch_size, *Feature_Map_px, 3, 85)。
                是每个批次数据的预测结果。
            sample_weight: update_state 方法的必备参数，即使不使用该参数，也必须在此
                进行定义，否则程序会报错。
            use_predictor: 一个布尔值。当使用测试盒 testcase 时，在每个单元测试中设置
            use_predictor=False，因为测试盒的 y_pred 是已经转换完成后的结果，不需要用
            predictor 再次转换。
        """

        # 先将模型输出进行转换。y_pred 形状为 (batch_size, *Feature_Map_px, 3, 85)。
        if use_predictor:
            y_pred = predictor(inputs=y_pred)

        # 先更新第一个状态量 showed_up_classes，更新该状态量不需要逐个图片处理。
        # 1. 先从标签中提取所有出现过的类别。
        # objectness_label 形状为 (batch_size, *Feature_Map_px, 3)。
        objectness_label = y_true[..., 0]

        # showed_up_categories_index_label 形状为
        # (batch_size, *Feature_Map_px, 3)，是一个布尔张量。
        showed_up_categories_index_label = tf.experimental.numpy.isclose(
            objectness_label, 1)

        # showed_up_categories_label 形状为 (batch_size, *Feature_Map_px, 3)。
        showed_up_categories_label = tf.argmax(y_true[..., 1: 81], axis=-1)

        # showed_up_categories_label 形状为 (x,)，里面存放的是出现过的类别编号，表示
        # 有 x 个类别出现在了这批标签中。
        showed_up_categories_label = showed_up_categories_label[
            showed_up_categories_index_label]

        # showed_up_categories_label 形状为 (1, x)。
        showed_up_categories_label = tf.reshape(showed_up_categories_label,
                                                shape=(1, -1))

        # 2. 从预测结果中提取所有出现过的类别，操作方法和上面的步骤 1 类似。
        # objectness_pred 形状为 (batch_size, *Feature_Map_px, 3)。
        objectness_pred = y_pred[..., 0]

        # classification_confidence_pred 形状为 (batch_size, *Feature_Map_px, 3)。
        classification_confidence_pred = tf.reduce_max(
            y_pred[..., 1: 81], axis=-1)

        # showed_up_categories_index_pred 形状为
        # (batch_size, *Feature_Map_px, 3)，是一个布尔张量。和 y_true 不同的地方在于，
        # 它需要大于 2 个置信度阈值，才认为是做出了预测，得出正确的布尔张量。
        showed_up_categories_index_pred = tf.logical_and(
            x=(objectness_pred > OBJECTNESS_THRESHOLD),
            y=(classification_confidence_pred >
               CLASSIFICATION_CONFIDENCE_THRESHOLD))

        # showed_up_categories_pred 形状为 (batch_size, *Feature_Map_px, 3)。
        showed_up_categories_pred = tf.argmax(y_pred[..., 1: 81], axis=-1)

        # showed_up_categories_pred 形状为 (y,)，里面存放的是出现过的类别编号，表示
        # 有 y 个类别出现在了这批预测结果中。
        showed_up_categories_pred = showed_up_categories_pred[
            showed_up_categories_index_pred]

        # showed_up_categories_pred 形状为 (1, y)。
        showed_up_categories_pred = tf.reshape(showed_up_categories_pred,
                                               shape=(1, -1))

        # showed_up_categories 形状为 (z,)，是一个 sparse tensor。对出现过的类别求
        # 并集，数量从 x,y 变为 z。
        showed_up_categories = tf.sets.union(showed_up_categories_pred,
                                             showed_up_categories_label)

        # 将 showed_up_categories 从 sparse tensor 转化为 tf.tensor。
        showed_up_categories = showed_up_categories.values

        # 更新状态量 showed_up_classes。
        # 遍历该 batch 中的每一个类别，如果该类别是第一次出现，则需要将其记录下来。
        for showed_up_category in showed_up_categories:
            if not showed_up_classes[showed_up_category]:
                showed_up_classes[showed_up_category].assign(True)

        # 下面更新另外 2 个状态量 latest_positive_bboxes 和
        # labels_quantity_per_image，需要逐个图片处理。
        batch_size = y_true.shape[0]

        # 步骤 1，遍历每一张图片预测结果及其对应的标签。
        for sample in range(batch_size):

            # one_label， one_pred 形状为 (*Feature_Map_px, 3, 85).
            one_label = y_true[sample]
            one_pred = y_pred[sample]

            # 步骤 2.1，对于标签，构造 3 个张量：positives_index_label，
            # positives_label 和 category_label。
            # objectness_one_label 形状为 (*Feature_Map_px, 3).
            objectness_one_label = one_label[..., 0]

            # positives_index_label 形状为 (*Feature_Map_px, 3)，是一个布尔张量。
            # 因为用了 isclose 函数，对标签要慎用 label smoothing，标签值可能不再为 1。
            positives_index_label = tf.experimental.numpy.isclose(
                objectness_one_label, 1)

            # positives_label 形状为 (*Feature_Map_px, 3, 85)，是标签正样本的信息，
            # 在不是正样本的位置，其数值为 -8。
            positives_label = tf.where(
                condition=positives_index_label[..., tf.newaxis],
                x=one_label, y=-8.)

            # category_label 形状为 (*Feature_Map_px, 3)，是标签正样本的类别编号，
            # 在不是正样本的位置，其数值为 0。因为这个 0 会和类别编号 0 发生混淆，所以下面
            # 要用 tf.where 再次进行转换。
            category_label = tf.argmax(positives_label[..., 1: 81], axis=-1,
                                       output_type=tf.dtypes.int32)

            # category_label 形状为 (*Feature_Map_px, 3)，是标签正样本的类别编号，
            # 在不是正样本的位置，其数值为 -8。
            category_label = tf.where(condition=positives_index_label,
                                      x=category_label, y=-8)

            # 步骤 2.2，对于预测结果，构造 3 个张量：positives_index_pred，
            # positives_pred 和 category_pred。

            # objectness_one_pred 形状为 (*Feature_Map_px, 3).
            objectness_one_pred = one_pred[..., 0]
            # classification_confidence_one_pred 形状为 (*Feature_Map_px, 3)。
            classification_confidence_one_pred = tf.reduce_max(
                one_pred[..., 1: 81], axis=-1)

            # positives_index_pred 形状为 (*Feature_Map_px, 3)，是一个布尔张量。
            positives_index_pred = tf.logical_and(
                x=(objectness_one_pred > OBJECTNESS_THRESHOLD),
                y=(classification_confidence_one_pred >
                   CLASSIFICATION_CONFIDENCE_THRESHOLD))

            # positives_pred 形状为 (*Feature_Map_px, 3, 85)，是预测结果正样本的信息，
            # 在不是正样本的位置，其数值为 -8。
            positives_pred = tf.where(
                condition=positives_index_pred[..., tf.newaxis],
                x=one_pred, y=-8.)

            # category_pred 形状为 (*Feature_Map_px, 3)，是预测结果正样本的类别编号，
            # 在不是正样本的位置，其数值为 0。因为这个 0 会和类别编号 0 发生混淆，所以下面
            # 要用 tf.where 再次进行转换。
            category_pred = tf.argmax(positives_pred[..., 1: 81], axis=-1,
                                      output_type=tf.dtypes.int32)

            # category_pred 形状为 (*Feature_Map_px, 3)，是预测结果正样本的类别编号，
            # 在不是正样本的位置，其数值为 -8。
            category_pred = tf.where(condition=positives_index_pred,
                                     x=category_pred, y=-8)

            # 步骤 3，遍历所有 80 个类别，更新另外 2 个状态值。
            # 对于每一个类别，可能会在 y_true, y_pred 中出现，也可能不出现。组合起来
            # 有 4 种情况，需要对这 4 种情况进行区分，更新状态值。
            for category in range(CLASSES):

                # category_bool_label 和 category_bool_pred 形状都为
                # (*Feature_Map_px, 3)，所有属于当前类别的 bboxes，其布尔值为 True。
                # 这也是把 category_label，category_pred 的非正样本位置设为 -8 的原
                # 因，避免和 category 0 发生混淆。
                category_bool_label = tf.experimental.numpy.isclose(
                    category_label, category)
                category_bool_pred = tf.experimental.numpy.isclose(
                    category_pred, category)

                # category_bool_any_label 和 category_bool_any_pred 是单个布尔值，
                # 用于判断 4 种情况。
                category_bool_any_label = tf.reduce_any(category_bool_label)
                category_bool_any_pred = tf.reduce_any(category_bool_pred)

                # 下面要分 4 种情况，更新状态量。
                # 情况 a ：标签和预测结果中，都没有该类别。无须更新状态。

                # 情况 b ：预测结果中没有该类别，但是标签中有该类别。
                # 对于预测结果，要提取置信度和 IoU，且置信度和 IoU 都为 0。
                # 对于标签，则提取该类别的标签数量即可。
                # scenario_b 是单个布尔值。
                scenario_b = tf.logical_and((~category_bool_any_pred),
                                            category_bool_any_label)

                # 情况 c ：预测结果中有该类别，标签没有该类别。
                # 对于预测结果，要提取置信度，而因为没有标签，IoU 为 0。
                # 对于标签，提取该类别的标签数量为 0 即可。
                # scenario_c 是单个布尔值。
                scenario_c = tf.logical_and(category_bool_any_pred,
                                            (~category_bool_any_label))

                # 情况 d ：预测结果和标签中都有该类别，此时要计算 IoU，再提取预测结果的
                # 置信度和 IoU，标签中则要提取标签数量。scenario_d 是单个布尔值。
                scenario_d = tf.logical_and(category_bool_any_pred,
                                            category_bool_any_label)

                # 只有在情况 b, c, d 时，才需要更新状态，所以先要判断是否处在情况
                # b, c, d 下。under_scenarios_bcd 是单个布尔值。
                under_scenarios_bc = tf.logical_or(scenario_b, scenario_c)
                under_scenarios_bcd = tf.logical_or(under_scenarios_bc,
                                                    scenario_d)

                # 在情况 b, c, d 时，更新状态量。
                if under_scenarios_bcd:
                    # 更新第二个状态量 labels_quantity_per_image，其形状为
                    # (CLASSES, latest_related_images)。
                    # one_image_category_labels_quantity 是一个整数，表示在一张图
                    # 片中，属于当前类别的标签 bboxes 数量。
                    one_image_category_labels_quantity = tf.where(
                        category_bool_label).shape[0]

                    # 如果某个类别没有在标签中出现，标签数量会是个 None，需要改为 0 。
                    if one_image_category_labels_quantity is None:
                        one_image_category_labels_quantity = 0

                    # 先把 labels_quantity_per_image 整体后移一位。
                    labels_quantity_per_image[category, 1:].assign(
                        labels_quantity_per_image[category, :-1])

                    # 把最近一个标签数量更新到 labels_quantity_per_image 的第 0 位。
                    labels_quantity_per_image[category, 0].assign(
                        one_image_category_labels_quantity)

                    # 最后更新第三个状态量 latest_positive_bboxes，形状为
                    # (CLASSES, latest_related_images, bboxes_per_image, 2)。
                    # 需要对 3 种情况 b,c,d 分别进行更新。

                    # 情况 b ：预测结果中没有该类别，但是标签中有该类别。
                    # 对于预测结果，要提取置信度和 IoU，且置信度和 IoU 都为 0。
                    if scenario_b:

                        # one_image_positive_bboxes 形状为 (BBOXES_PER_IMAGE, 2)。
                        one_image_positive_bboxes = tf.zeros(
                            shape=(BBOXES_PER_IMAGE, 2))

                    # 情况 c ：预测结果中有该类别，标签没有该类别。
                    # 对于预测结果的状态，要提取置信度，而因为没有标签，IoU 为 0。
                    elif scenario_c:
                        # scenario_c_positives_pred 形状为
                        # (scenario_c_bboxes, 85)。
                        scenario_c_positives_pred = positives_pred[
                            category_bool_pred]

                        # scenario_c_class_confidence_pred 形状为
                        # (scenario_c_bboxes,)。
                        scenario_c_class_confidence_pred = tf.reduce_max(
                                scenario_c_positives_pred[:, 1: 81], axis=-1)

                        scenario_c_bboxes = (
                            scenario_c_class_confidence_pred.shape[0])

                        if scenario_c_bboxes is None:
                            scenario_c_bboxes = 0

                        # 如果 scenario_c_bboxes 数量少于规定的数量，则进行补零。
                        if scenario_c_bboxes < BBOXES_PER_IMAGE:
                            # scenario_c_paddings 形状为 (1, 2)。
                            scenario_c_paddings = tf.constant(
                                (0, (BBOXES_PER_IMAGE - scenario_c_bboxes)),
                                shape=(1, 2))

                            # one_image_positive_bboxes 形状为
                            # (BBOXES_PER_IMAGE,)。
                            one_image_positive_bboxes = tf.pad(
                                tensor=scenario_c_class_confidence_pred,
                                paddings=scenario_c_paddings,
                                mode='CONSTANT', constant_values=0)

                        # 如果 scenario_c_bboxes 数量大于等于规定的数量，则应该先按
                        # 类别置信度从大到小的顺序进行排序，然后保留规定的数量 bboxes。
                        else:
                            # scenario_c_sorted_pred 形状为
                            # (BBOXES_PER_IMAGE,)。
                            scenario_c_sorted_pred = tf.sort(
                                scenario_c_class_confidence_pred,
                                direction='DESCENDING')

                            # one_image_positive_bboxes 形状为
                            # (BBOXES_PER_IMAGE,)。
                            one_image_positive_bboxes = (
                                scenario_c_sorted_pred[: BBOXES_PER_IMAGE])

                        # scenario_c_ious_pred 形状为 (BBOXES_PER_IMAGE,)。
                        scenario_c_ious_pred = tf.zeros_like(
                            one_image_positive_bboxes)

                        # one_image_positive_bboxes 形状为 (BBOXES_PER_IMAGE, 2)。
                        one_image_positive_bboxes = tf.stack(
                            values=[one_image_positive_bboxes,
                                    scenario_c_ious_pred], axis=1)

                    # 情况 d ：预测结果和标签中都有该类别，此时要计算 IoU，再提取预测结果
                    # 的置信度和 IoU，标签中则要提取标签数量。scenario_d 是单个布尔值。
                    else:
                        # 1. bboxes_iou_pred 形状为 (*Feature_Map_px, 3, 4)。
                        bboxes_iou_pred = tf.where(
                            condition=category_bool_pred[..., tf.newaxis],
                            x=positives_pred[..., -4:], y=0.)

                        # 2. 构造 bboxes_category_label， 形状为
                        # (scenario_d_bboxes_label, 4)。
                        bboxes_category_label = positives_label[..., -4:][
                            category_bool_label]

                        # bboxes_area_label 形状为 (scenario_d_bboxes_label,)，
                        # 是当前类别中，各个 bbox 的面积。
                        bboxes_area_label = (bboxes_category_label[:, -1] *
                                             bboxes_category_label[:, -2])

                        # 把标签的 bboxes 按照面积从小到大排序。
                        # sort_by_area 形状为 (scenario_d_bboxes_label,)
                        sort_by_area = tf.argsort(values=bboxes_area_label,
                                                  axis=0, direction='ASCENDING')

                        # 3. 构造 sorted_bboxes_label， 形状为
                        # (scenario_d_bboxes_label, 4)。
                        sorted_bboxes_label = tf.gather(
                            params=bboxes_category_label,
                            indices=sort_by_area, axis=0)

                        # 4. 用 one_image_positive_bboxes 记录下新预测的且命中标签的
                        # bboxes，直接设置其为空，后续用 concat 方式添加新的 bboxes。
                        one_image_positive_bboxes = tf.zeros(
                            shape=(BBOXES_PER_IMAGE, 2))

                        # 用 new_bboxes_quantity 作为标识 flag，每向
                        # one_image_positive_bboxes 增加一个 bbox 信息，则变大 1.
                        new_bboxes_quantity = 0

                        # 5. 遍历 sorted_bboxes_label。
                        for bbox_info in sorted_bboxes_label:

                            # carried_over_shape 形状为 (*Feature_Map_px, 3, 4)
                            carried_over_shape = tf.ones_like(bboxes_iou_pred)

                            # 5.1 构造 bbox_iou_label，
                            # 其形状为 (*Feature_Map_px, 3, 4)。
                            bbox_iou_label = carried_over_shape * bbox_info

                            # 5.2 ious_category 形状为 (*Feature_Map_px, 3)。
                            ious_category = iou_calculator(
                                label_bbox=bbox_iou_label,
                                prediction_bbox=bboxes_iou_pred)

                            # max_iou_category 是一个标量，表示当前类别所有 bboxes，
                            # 计算得到的最大 IoU。
                            max_iou_category = tf.reduce_max(ious_category)

                            # 5.3 当最大 IoU 大于 0.5 时，则认为预测的 bbox 命中了该
                            # 标签，需要把置信度和 IoU 记录到 category_new_bboxes 中。
                            if tf.logical_and(
                                    (max_iou_category > 0.5),
                                    (new_bboxes_quantity < BBOXES_PER_IMAGE)):
                                # 记录 new_bboxes_quantity，当达到设定的固定数量后，
                                # 停止记录新的 bboxes。
                                new_bboxes_quantity += 1

                                # max_iou_position 形状为 (*Feature_Map_px, 3)，
                                # 是一个布尔张量，仅最大 IoU 位置为 True。
                                max_iou_position = (
                                    tf.experimental.numpy.isclose(
                                        ious_category, max_iou_category))

                                # max_iou_bbox_pred 形状为 (1, 85)，是预测结果中
                                # IoU 最大的那个 bbox。
                                max_iou_bbox_pred = positives_pred[
                                    max_iou_position]

                                # max_iou_bbox_confidence 是一个标量型张量。
                                max_iou_bbox_class_confidence = (
                                    tf.reduce_max(max_iou_bbox_pred[0, 1: 81]))

                                # new_bbox 是一个元祖，包含类别置信度和 IoU。
                                new_bbox = (max_iou_bbox_class_confidence,
                                            max_iou_category)

                                # new_bbox 形状为 (1, 2)。
                                new_bbox = tf.ones(shape=(1, 2)) * new_bbox

                                # 记录这个命中标签的 bbox 信息。append_new_bboxes
                                # 形状为 (BBOXES_PER_IMAGE + 1, 2)。
                                append_new_bboxes = tf.concat(
                                    values=[one_image_positive_bboxes,
                                            new_bbox], axis=0)

                                # 5.3.1 记录到 one_image_positive_bboxes， 形状为
                                # (BBOXES_PER_IMAGE, 2)。
                                one_image_positive_bboxes = (
                                    append_new_bboxes[-BBOXES_PER_IMAGE:])

                                # 5.3.2 需要将该 bbox 从 bboxes_iou_pred
                                # 中移除，再进行后续的 IoU 计算。remove_max_iou_bbox
                                # 形状为 (*Feature_Map_px, 3, 1)，在最大 IoU 的位
                                # 置为 True，其它为 False。
                                remove_max_iou_bbox = max_iou_position[
                                    ..., tf.newaxis]

                                # bboxes_iou_pred 形状为 (*Feature_Map_px,
                                # 3, 4)。把被去除的 bbox 替换为 0。
                                bboxes_iou_pred = tf.where(
                                    condition=remove_max_iou_bbox,
                                    x=0., y=bboxes_iou_pred)

                        # 6. 遍历 sorted_bboxes_label 完成之后，处理
                        # bboxes_iou_pred 中剩余的 bboxes。

                        # left_bboxes_sum 形状为 (*Feature_Map_px, 3)，是剩
                        # 余的没有命中标签的 bboxes。
                        # 下面用求和，是为了确定该 bbox 中是否有物体。如果一个
                        # bbox 中没有物体，那么它的中心点坐标，高度宽度这 4 个
                        # 参数之和依然等于 0。
                        left_bboxes_sum = tf.math.reduce_sum(
                            bboxes_iou_pred, axis=-1)

                        # left_bboxes_bool 形状为 (*Feature_Map_px, 3)，
                        # 是一个布尔张量，剩余 bboxes 位置为 True。
                        left_bboxes_bool = (left_bboxes_sum > 0)

                        # left_bboxes_pred 形状为 (left_bboxes_quantity, 85)。
                        left_bboxes_pred = positives_pred[left_bboxes_bool]

                        # left_bboxes_confidence_pred 是剩余 bboxes 的类别
                        # 置信度，形状为 (left_bboxes_quantity,)。
                        left_bboxes_confidence_pred = tf.reduce_max(
                            left_bboxes_pred[:, 1: 81], axis=-1)

                        # left_bboxes_quantity 是一个标量型张量。
                        left_bboxes_quantity = left_bboxes_pred.shape[0]

                        if left_bboxes_quantity is None:
                            left_bboxes_quantity = 0

                        # 把没有命中标签的正样本 bboxes 也记录下来。
                        if tf.math.logical_and(
                                (left_bboxes_quantity > 0),
                                (new_bboxes_quantity < BBOXES_PER_IMAGE)):

                            # scenario_d_bboxes 是一个标量型张量。
                            scenario_d_bboxes = (new_bboxes_quantity +
                                                 left_bboxes_quantity)

                            # 6.1 scenario_d_bboxes > BBOXES_PER_IMAGE，需
                            # 要对剩余的 bboxes，按类别置信度进行排序。
                            if scenario_d_bboxes > BBOXES_PER_IMAGE:
                                # left_bboxes_sorted_confidence 形状为
                                # (left_bboxes_quantity,)。
                                left_bboxes_sorted_confidence = tf.sort(
                                    left_bboxes_confidence_pred,
                                    direction='DESCENDING')

                                # vacant_seats 是一个整数，表示还有多少个空位，
                                # 可以用于填充剩余的 bboxes。
                                vacant_seats = (
                                    BBOXES_PER_IMAGE - new_bboxes_quantity)

                                # left_bboxes_confidence_pred 形状为
                                # (vacant_seats,)。
                                left_bboxes_confidence_pred = (
                                    left_bboxes_sorted_confidence[
                                        : vacant_seats])

                            # left_bboxes_ious_pred 形状为 (vacant_seats,)，
                            # 或者是 (left_bboxes_quantity,)。
                            left_bboxes_ious_pred = tf.zeros_like(
                                left_bboxes_confidence_pred)

                            # left_positive_bboxes_pred 形状为
                            # (left_bboxes_quantity, 2)。
                            left_positive_bboxes_pred = tf.stack(
                                values=[left_bboxes_confidence_pred,
                                        left_bboxes_ious_pred], axis=1)

                            # 记录剩余 bboxes 信息。append_left_bboxes
                            # 形状为 (BBOXES_PER_IMAGE +
                            # left_bboxes_quantity, 2)。
                            append_left_bboxes = tf.concat(
                                values=[one_image_positive_bboxes,
                                        left_positive_bboxes_pred],
                                axis=0)

                            # one_image_positive_bboxes，形状为
                            # (BBOXES_PER_IMAGE, 2)。
                            one_image_positive_bboxes = (
                                append_left_bboxes[-BBOXES_PER_IMAGE:])

                    # 更新最后一个状态量 latest_positive_bboxes。 形状为 (CLASSES,
                    # LATEST_RELATED_IMAGES, BBOXES_PER_IMAGE, 2)。
                    latest_positive_bboxes[category, 1:].assign(
                        latest_positive_bboxes[category, :-1])

                    # latest_positive_bboxes 形状为 (CLASSES,
                    # LATEST_RELATED_IMAGES, BBOXES_PER_IMAGE, 2)。
                    latest_positive_bboxes[category, 0].assign(
                        one_image_positive_bboxes)

    def result(self):
        """对于当前所有已出现类别，使用状态值 state，计算 mean average precision。"""

        # 只在 P3 时计算 AP，因为 P5, P4 实际上是无效计算，跳开 P5, P4 还可以节省时间。
        # 和 YOLOv4-CSP 等 YOLO 系列模型配合使用时，注意要将 3 个输出分别命名为 p5,
        # p4, p3，它们会自动和 AP 指标连接，得到指标名字 self.name 为 'p3_AP' 等。
        if self.name == 'p3_AP':

            # 不能直接使用 tf.Variable 进行索引，需要将其转换为布尔张量。
            # showed_up_classes 形状为 (CLASSES,)。
            showed_up_classes_tensor = tf.convert_to_tensor(
                showed_up_classes, dtype=tf.bool)

            # average_precision_per_iou 形状为 (10,)。
            average_precision_per_iou = tf.zeros(shape=(10,))
            # 把 10 个不同 IoU 阈值情况下的 AP，放入张量 average_precision_per_iou
            # 中，然后再求均值。
            for iou_threshold in np.linspace(0.5, 0.95, num=10):

                # average_precisions 形状为 (80,)，存放的是每一个类别的 AP。
                average_precisions = tf.zeros(shape=(CLASSES,))
                # 对所有出现过的类别，将其 AP 放入 average_precisions 中，然后再求均值。
                for category in range(CLASSES):

                    # 只使用出现过的类别计算 AP。
                    if showed_up_classes[category]:
                        # 1. 计算 recall_precisions。
                        recall_precisions = tf.ones(shape=(1,))
                        true_positives = tf.constant(0., shape=(1,))
                        false_positives = tf.constant(0., shape=(1,))

                        # 下面按照类别置信度从大到小的顺序，对 bboxes 进行排序。
                        # positive_bboxes_category 形状为
                        # (LATEST_RELATED_IMAGES, BBOXES_PER_IMAGE, 2)
                        positive_bboxes_category = latest_positive_bboxes[
                            category]

                        # positive_bboxes_category 形状为
                        # (LATEST_RELATED_IMAGES * BBOXES_PER_IMAGE, 2)
                        positive_bboxes_category = tf.reshape(
                            positive_bboxes_category, shape=(-1, 2))

                        # confidence_category 形状为
                        # (LATEST_RELATED_IMAGES * BBOXES_PER_IMAGE,)。
                        confidence_category = positive_bboxes_category[:, 0]

                        # sorted_classification_confidence 形状为
                        # (LATEST_RELATED_IMAGES * BBOXES_PER_IMAGE,)。
                        sorted_classification_confidence = tf.argsort(
                            values=confidence_category,
                            axis=0, direction='DESCENDING')

                        # sorted_bboxes_category 形状为
                        # (LATEST_RELATED_IMAGES * BBOXES_PER_IMAGE, 2)。
                        sorted_bboxes_category = tf.gather(
                            params=positive_bboxes_category,
                            indices=sorted_classification_confidence, axis=0)

                        # 一个奇怪的事情是，使用 for bbox in sorted_bboxes_category，
                        # 它将不允许对 recall_precisions 使用 tf.concat。
                        # 下面更新 recall_precisions。
                        for i in range(len(sorted_bboxes_category)):
                            bbox = sorted_bboxes_category[i]
                            # sorted_bboxes_category 中，有一些是空的 bboxes，是既
                            # 没有标签，也没有预测结果。当遇到这些 bboxes 时，说明已经遍
                            # 历完预测结果，此时应跳出循环。空的 bboxes 类别置信度为 0.
                            bbox_classification_confidence = bbox[0]

                            if bbox_classification_confidence > 0:
                                bbox_iou = bbox[1]
                                # 根据当前的 iou_threshold，判断该 bbox 是否命中标签。
                                if bbox_iou > iou_threshold:
                                    true_positives += 1
                                    # 如果增加了一个 recall ，则记录下来。
                                    recall_increased = True
                                else:
                                    false_positives += 1
                                    recall_increased = False

                                # 计算精度 precision。
                                precision = true_positives / (true_positives +
                                                              false_positives)

                                # recall_precisions 形状为 (x,)。如果有新增加了一个
                                # recall，则增加一个新的精度值。反之如果 recall 没有
                                # 增加，则把当前的精度值更新即可。
                                recall_precisions = tf.cond(
                                    pred=recall_increased,
                                    true_fn=lambda: tf.concat(
                                        values=[recall_precisions, precision],
                                        axis=0),
                                    false_fn=lambda: tf.concat(
                                        values=[recall_precisions[:-1],
                                                precision], axis=0))

                        # 2. 计算当前类别的 AP。使用累加多个小梯形面积的方式来计算 AP。

                        # labels_quantity 是当前类别中，所有标签的总数。
                        labels_quantity = tf.math.reduce_sum(
                            labels_quantity_per_image[category])

                        # update_state 方法中区分了 a,b,c,d 共 4 种情况，scenario_d
                        # 属于下面这种，即有预测结果和标签，需要计算 AP 的情况。
                        # 如果有标签，即 labels_quantity > 0，要计算 AP。
                        if labels_quantity > 0:

                            # trapezoid_height 是每一个小梯形的高度。
                            # 注意！！！如果没有标签也计算小梯形高度，trapezoid_height
                            # 将会是 inf，并最终导致 NaN。所以要设置
                            # labels_quantity > 0.
                            trapezoid_height = 1 / labels_quantity

                            # accumulated_edge_length 是每一个小梯形的上下边长总和。
                            # accumulated_edge_length = 0.
                            accumulated_edge_length = tf.constant(
                                0., dtype=tf.float32)

                            # recalls 是总的 recall 数量。因为第 0 位并不是真正的
                            # recall，所以要减去 1.
                            recalls = len(recall_precisions) - 1

                            if recalls == 0:
                                # scenario_b 是有标签但是没有预测结果，包括在这种情况
                                # recalls==0，累计的梯形面积应该等于 0，AP 也将等于0。
                                accumulated_area_trapezoid = tf.constant(
                                    0, dtype=tf.float32)

                            else:
                                for i in range(recalls):
                                    top_edge_length = recall_precisions[i]
                                    bottom_edge_length = recall_precisions[
                                        i + 1]

                                    accumulated_edge_length += (
                                            top_edge_length +
                                            bottom_edge_length)

                                # 计算梯形面积：(上边长 + 下边长) * 梯形高度 / 2 。
                                accumulated_area_trapezoid = (
                                    accumulated_edge_length *
                                    trapezoid_height) / 2

                        # 而如果没有标签，则 average_precision=0。
                        # accumulated_area_trapezoid 就是当前类别的
                        # average_precision。scenario_c 属于这种情况。
                        else:
                            accumulated_area_trapezoid = tf.constant(
                                0, dtype=tf.float32)

                        # 构造索引 category_index，使它指向当前类别。
                        category_index = np.zeros(shape=(CLASSES,))
                        category_index[category] = 1
                        # category_index 形状为 (CLASSES,)。
                        category_index = tf.convert_to_tensor(category_index,
                                                              dtype=tf.bool)

                        # average_precisions 形状为 (CLASSES,)。
                        average_precisions = tf.where(
                            condition=category_index,
                            x=accumulated_area_trapezoid[tf.newaxis],
                            y=average_precisions)

                # 把出现过的类别过滤出来，用于计算 average_precision。
                # average_precision_showed_up_categories 形状为 (x,)，即一共有 x
                # 个类别出现过，需要参与计算 AP。
                average_precision_showed_up_categories = average_precisions[
                    showed_up_classes_tensor]

                # average_precision_over_categories 形状为 (1,)。
                average_precision_over_categories = tf.math.reduce_mean(
                    average_precision_showed_up_categories, keepdims=True)

                # average_precision_per_iou 形状始终保持为 (10,)。
                average_precision_per_iou = tf.concat(
                    values=[average_precision_per_iou[1:],
                            average_precision_over_categories], axis=0)

            mean_average_precision = tf.math.reduce_mean(
                average_precision_per_iou)

        # 虽然跳开了 P5, P4，但是还需要给它们设置一个浮点数，否则会报错。
        else:
            mean_average_precision = 0.

        return mean_average_precision

    def reset_state(self):
        """每个 epoch 开始时，需要重新把状态初始化。"""
        latest_positive_bboxes.assign(
            tf.zeros_like(latest_positive_bboxes))

        labels_quantity_per_image.assign(
            tf.zeros_like(labels_quantity_per_image))

        showed_up_classes.assign(tf.zeros_like(showed_up_classes))

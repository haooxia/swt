import os
from typing import TypeVar, NamedTuple, List, Optional, Tuple
import cv2
import numpy as np
from scipy.spatial import ConvexHull
import sys
from .arg_parser import SwtArgParser
from .id_colors import build_colormap


Image = np.ndarray
GradientImage = np.ndarray
Position = NamedTuple('Position', [('x', int), ('y', int)])  # 命名元组
Stroke = NamedTuple('Stroke', [('x', int), ('y', int), ('width', float)])
Ray = List[Position]
Component = List[Position]
ImageOrValue = TypeVar('ImageOrValue', float, Image)
Gradients = NamedTuple('Gradients', [('x', GradientImage), ('y', GradientImage)])


def gamma(x: ImageOrValue, coeff: float=2.2) -> ImageOrValue:
    """
    Applies a gamma transformation to the input.

    :param x: The value to transform.
    :param coeff: The gamma coefficient to use.
    :return: The transformed value.
    """
    return x ** (1./coeff)


def gleam(im: Image, gamma_coeff: float=2.2) -> Image:  # gleam灰度变换
    """
    Implements Gleam grayscale conversion from
    Kanan & Cottrell 2012: Color-to-Grayscale: Does the Method Matter in Image Recognition?
    http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029740

    :param im: The image to convert.
    :param gamma_coeff: The gamma coefficient to use.
    :return: The grayscale converted image.
    """
    im = gamma(im, gamma_coeff)
    im = np.mean(im, axis=2)
    return np.expand_dims(im, axis=2)


def open_grayscale(path: str) -> Image:
    """
    Opens an image and converts it to grayscale.

    :param path: The image to open.
    :return: The grayscale image.
    """
    im = cv2.imread(path, cv2.IMREAD_COLOR) # 彩色读入
    im = im.astype(np.float32) / 255. # -> [0,1]
    return gleam(im) # to gray


def get_edges(im: Image, lo: float=175, hi: float=220, window: int=3) -> Image:
    """
    Detects edges in the image by applying a Canny edge detector.

    :param im: The image.
    :param lo: The lower threshold.
    :param hi: The higher threshold.
    :param window: The window (aperture) size.
    :return: The edges.
    """
    # OpenCV's Canny detector requires 8-bit inputs.
    im = (im * 255.).astype(np.uint8) # [0, 255]
    edges = cv2.Canny(im, lo, hi, apertureSize=window)
    # Note that the output is either 255 for edges or 0 for other pixels.
    # Conversion to float wastes space, but makes the return value consistent
    # with the other methods.
    return edges.astype(np.float32) / 255.


def get_gradients(im: Image) -> Gradients:
    """
    Obtains the image gradients by means of a 3x3 Scharr filter.

    :param im: The image to process.
    :return: The image gradients.
    """
    # In 3x3, Scharr is a more correct choice than Sobel. For higher
    # dimensions, Sobel should be used.
    grad_x = cv2.Scharr(im, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(im, cv2.CV_64F, 0, 1)
    return Gradients(x=grad_x, y=grad_y)


def get_gradient_directions(g: Gradients) -> Image:
    """
    Obtains the gradient directions.

    :param g: The gradients.
    :return: An image of the gradient directions.
    """
    return np.arctan2(g.y, g.x)


def apply_swt(im: Image, edges: Image, gradients: Gradients, dark_on_bright: bool=True) -> Image:
    # SWT算法，对图像进行了文本边缘宽度变换
    """
    :param im: The image
    :param edges: The edges of the image.
    :param gradients: The gradients of the image.
    :param dark_on_bright: Enables dark-on-bright text detection. 使用白色背景
    :return: The transformed image.
    """
    swt = np.squeeze(np.ones_like(im)) * np.Infinity  # 创建了一个和输入图像大小相同的初始空白图像swt，初始值设置为正无穷。
    # For each pixel, let's obtain the normal direction of its gradient.
    # 计算梯度图像 gradients 的模值，并确保避免除以零的情况
    norms = np.sqrt(gradients.x ** 2 + gradients.y ** 2) # 梯度模值代表梯度向量的长度 
    norms[norms == 0] = 1
    inv_norms = 1. / norms # 梯度模值倒数代表长度的倒数
    
    # 这一部分代码计算了每个像素点在横向（x方向）上的梯度分量乘以对应位置的梯度模值的倒数。
    # 这个操作的目的是将每个像素点的横向梯度分量按照其梯度模值的倒数进行缩放。
    # 这样做的结果是，每个像素点的横向梯度分量被归一化为单位长度，以便在计算中更好地反映梯度方向
    # 总的来说：这里对梯度进行了归一化，目的是强调梯度的方向，忽略梯度的幅度
    directions = Gradients(x=gradients.x * inv_norms, y=gradients.y * inv_norms)

    # 初始化一个空的 rays 列表，用于存储找到的射线
    # We keep track of all the rays found in the image.
    rays = []
    
    # 遍历图像中的每个像素，如果该像素处在边缘上（通过判断 edges 值是否小于0.5），则调用 swt_process_pixel 函数处理该像素，并将返回的射线添加到 rays 列表中。
    # Find a pixel that lies on an edge.
    edge_cnt = 0
    ray_cnt = 0
    height, width = im.shape[0:2]
    for y in range(height):
        for x in range(width):
            # Edges are either 0. or 1.
            if edges[y, x] < .5: # 0 means not edge(edge is white)
                continue
            edge_cnt += 1
            ray = swt_process_pixel(Position(x=x, y=y), edges, directions, out=swt, dark_on_bright=dark_on_bright)
            if ray:
                ray_cnt += 1
                rays.append(ray)
    print('edge cnt:', edge_cnt)
    print('ray cnt:', ray_cnt)

    # Multiple rays may cross the same pixel and each pixel has the smallest
    # stroke width of those.
    # A problem are corners like the edge of an L. Here, two rays will be found,
    # both of which are significantly longer than the actual width of each
    # individual stroke. To mitigate, we will visit each pixel on each ray and
    # take the median stroke length over all pixels on the ray.
    # TODO TODO 访问每条射线上的每个像素，并获取射线上所有像素的中值笔画长度
    for ray in rays:
        # 计算射线上所有像素的边缘宽度的中位数（median），使用列表推导式 [swt[p.y, p.x] for p in ray] 获取射线上所有像素的边缘宽度
        # 将该像素的边缘宽度更新为中位数（median）和当前SWT图像中对应像素的边缘宽度的较小值
        median = np.median([swt[p.y, p.x] for p in ray])
        for p in ray:
            swt[p.y, p.x] = min(median, swt[p.y, p.x])
    # TODO TODO

    swt[swt == np.Infinity] = 0
    return swt


def swt_process_pixel(pos: Position, edges: Image, directions: Gradients, out: Image, dark_on_bright: bool=True) -> Optional[Ray]:
    """
    返回射线上像素位置的列表ray
    如果成功找到了完整的文本边缘 && 边缘的另一侧符合条件（梯度方向与起始方向近似相反
    Obtains the stroke width starting from the specified position.
    :param pos: The starting point
    :param edges: The edges.
    :param directions: The normalized gradients
    :param out: The output image.
    :param dark_on_bright: Enables dark-on-bright text detection.
    """
    # Keep track of the image dimensions for boundary tests.
    height, width = edges.shape[0:2]

    gradient_direction = -1 if dark_on_bright else 1

    # Starting from the current pixel we will shoot a ray into the direction
    # of the pixel's gradient and keep track of all pixels in that direction
    # that still lie on an edge.
    # 初始化一个包含起始点的射线列表 ray
    ray = [pos]
    # 获取当前像素梯度方向的步进量
    # Obtain the direction to step into
    dir_x = directions.x[pos.y, pos.x]
    dir_y = directions.y[pos.y, pos.x]

    # Since some pixels have no gradient, normalization of the gradient
    # is a division by zero for them, resulting in NaN. These values
    # should not bother us since we explicitly tested for an edge before.
    assert not (np.isnan(dir_x) or np.isnan(dir_y))

    # Traverse the pixels along the direction.
    prev_pos = Position(x=-1, y=-1)
    steps_taken = 0
    while True:
        # Advance to the next pixel on the line.
        steps_taken += 1
        cur_x = int(np.floor(pos.x + gradient_direction * dir_x * steps_taken))
        cur_y = int(np.floor(pos.y + gradient_direction * dir_y * steps_taken))
        cur_pos = Position(x=cur_x, y=cur_y)
        if cur_pos == prev_pos:
            continue
        prev_pos = Position(x=cur_x, y=cur_y)
        # If we reach the edge of the image without crossing a stroke edge,
        # we discard the result.
        if not ((0 <= cur_x < width) and (0 <= cur_y < height)):
            return None
        # The point is either on the line or the end of it, so we register it.
        ray.append(cur_pos)
        # If that pixel is not an edge, we are still on the line and
        # need to continue scanning.
        if edges[cur_y, cur_x] < .5:  # TODO: Test for image boundaries here 如果此点不是边缘点 则在该方向上拉长步子
            continue
        # If this edge is pointed in a direction approximately opposite of the
        # one we started in, it is approximately parallel. This means we
        # just found the other side of the stroke.
        # The original paper suggests the gradients need to be opposite +/- PI/6.
        # Since the dot product is the cosine of the enclosed angle and
        # cos(pi/6) = 0.8660254037844387, we can discard all values that exceed
        # this threshold.
        cur_dir_x = directions.x[cur_y, cur_x]
        cur_dir_y = directions.y[cur_y, cur_x]
        dot_product = dir_x * cur_dir_x + dir_y * cur_dir_y # 向量点击用来衡量二者之间的相似度/夹角
        # 注意，两个向量的模应该都是1，那么向量点积即为夹角余弦值
        # print(np.sqrt(dir_x * dir_x + dir_y * dir_y))
        # print(np.sqrt(cur_dir_x * cur_dir_x + cur_dir_y * cur_dir_y))

        # if dot_product >= -0.866: # 夹角在[0, 5pi/6] 扔掉
        # if dot_product >= -0.809: # [0, 4pi/5]
        # if dot_product >= -0.707: # cos(3*pi/4) 倘若两个向量夹角余弦值大于该值（夹角在[0,4pi/5]）则扔掉
        if dot_product >= -0.5: # cos(2*pi/3)
        # if dot_product >= -0.3: # cos(2*pi/3)
            return None
        # Paint each of the pixels on the ray with their determined stroke width
        stroke_width = np.sqrt((cur_x - pos.x) * (cur_x - pos.x) + (cur_y - pos.y) * (cur_y - pos.y))
        for p in ray:
            out[p.y, p.x] = min(stroke_width, out[p.y, p.x]) # 存下每一点的宽度值，后续用于体现亮度 TODO 此处为什么要保留重复结果的较小值呢？
        return ray



def connected_components(swt: Image, threshold: float=3.) -> Tuple[Image, List[Component]]:
    """
    Applies Connected Components labeling to the transformed image using a flood-fill algorithm.

    :param swt: The Stroke Width transformed image.
    :param threshold: The Stroke Width ratio below which two strokes are considered the same.
    :return: The map of labels.
    """
    height, width = swt.shape[0:2]
    labels = np.zeros_like(swt, dtype=np.uint32)
    next_label = 0
    components = []  # List[Component]
    for y in range(height):
        for x in range(width):
            stroke_width = swt[y, x]
            if (stroke_width <= 0) or (labels[y, x] > 0):
                continue
            next_label += 1
            neighbor_labels = [Stroke(x=x, y=y, width=stroke_width)]
            component = []
            while len(neighbor_labels) > 0:
                neighbor = neighbor_labels.pop()
                npos, stroke_width = Position(x=neighbor.x, y=neighbor.y), neighbor.width
                if not ((0 <= npos.x < width) and (0 <= npos.y < height)):
                    continue
                # If the current pixel was already labeled, skip it.
                n_label = labels[npos.y, npos.x]
                if n_label > 0:
                    continue
                # We associate pixels based on their stroke width. If there is no stroke, skip the pixel.
                n_stroke_width = swt[npos.y, npos.x]
                if n_stroke_width <= 0:
                    continue
                # We consider this point only if it is within the acceptable threshold and in the initial test
                # (i.e. when visiting a new stroke), the ratio is 1.
                # If we succeed, we can label this pixel as belonging to the same group. This allows for
                # varying stroke widths due to e.g. perspective distortion or elaborate fonts.
                if (stroke_width/n_stroke_width >= threshold) or (n_stroke_width/stroke_width >= threshold):
                    continue
                labels[npos.y, npos.x] = next_label
                component.append(npos)
                # From here, we're going to expand the new neighbors.
                neighbors = {Stroke(x=npos.x - 1, y=npos.y - 1, width=n_stroke_width),
                             Stroke(x=npos.x, y=npos.y - 1, width=n_stroke_width),
                             Stroke(x=npos.x + 1, y=npos.y - 1, width=n_stroke_width),
                             Stroke(x=npos.x - 1, y=npos.y, width=n_stroke_width),
                             Stroke(x=npos.x + 1, y=npos.y, width=n_stroke_width),
                             Stroke(x=npos.x - 1, y=npos.y + 1, width=n_stroke_width),
                             Stroke(x=npos.x, y=npos.y + 1, width=n_stroke_width),
                             Stroke(x=npos.x + 1, y=npos.y + 1, width=n_stroke_width)}
                neighbor_labels.extend(neighbors)
            if len(component) > 0:
                components.append(component)
    return labels, components


def minimum_area_bounding_box(points: np.ndarray) -> np.ndarray:
    """
    Determines the minimum area bounding box for the specified set of points.

    :param points: The point coordinates.
    :return: The coordinates of the bounding box.
    """
    # The minimum area bounding box is aligned with at least one
    # edge of the convex hull. (TODO: Proof?)
    # This reduces the number of orientations we have to try.
    hull = ConvexHull(points)
    for i in range(len(hull.vertices)-1):
        # Select two vertex pairs and obtain their orientation to the X axis.
        a = points[hull.vertices[i]]
        b = points[hull.vertices[i + 1]]
        # TODO: Find orientation. Note that sine = abs(cross product) and cos = dot product of two vectors.
        print(a, b)
    return points


def discard_non_text(swt: Image, labels: Image, components: List[Component]) -> Tuple[Image, List[Component]]:
    """
    Discards components that are likely not text.
    
    :param swt: The stroke-width transformed image.
    :param labels: The labeled components.
    :param components: A list of each component with all its pixels.
    :return: The filtered labels and components.
    """
    invalid_components = []  # type: List[Component]
    for component in components:
        # If the variance of the stroke widths in the component is more than
        # half the average of the stroke widths of that component, it is considered invalid.
        average_stroke = np.mean([swt[p.y, p.x] for p in component])
        variance = np.var([swt[p.y, p.x] for p in component])
        if variance > .5*average_stroke:
            invalid_components.append(component)
            continue
        # Natural scenes may create very long, yet narrow components. We prune
        # these based on their aspect ratio.
        points = np.array([[p.x, p.y] for p in component], dtype=np.uint32)
        minimum_area_bounding_box(points)
        print(variance)
    return labels, components


def main():

    parser = SwtArgParser()
    args = parser.parse_args()

    if not os.path.exists(args.image):
        parser.error('Image file does not exist: {}'.format(args.image))
    # Open the image and obtain a grayscale representation.
    im = open_grayscale(args.image)  # Magic numbers hidden in arguments

    # Find the edges in the image and the gradients.
    edges = get_edges(im)  # Magic numbers hidden in arguments

    cv2.imwrite('canny_edge.png', edges*255)

    gradients = get_gradients(im)  # Magic numbers hidden in arguments 获取xy梯度

    # TODO: Gradient directions are only required for checking if two edges are in opposing directions. We can use the gradients directly.
    # Obtain the gradient directions. Due to symmetry, we treat opposing
    # directions as the same (e.g. 180° as 0°, 135° as 45°, etc.).
    # theta = get_gradient_directions(gradients)
    # theta = np.abs(theta)

    # Apply the Stroke Width Transformation.
    swt = apply_swt(im, edges, gradients, not args.bright_on_dark)
    # TODO swt返回的是0和一些宽度 亮度一样
    # for i in range(len(swt)):
        # print(swt[i])
    # cv2.imshow('swt_before', swt)

    #####################################

    # Apply Connected Components labelling
    # :return: The map of labels.
    labels, components = connected_components(swt)  # TODO: Magic numbers hidden in arguments
    # Discard components that are likely not text
    # TODO: labels, components = discard_non_text(swt, labels, components)

    labels = labels.astype(np.float32) / labels.max() # [0,1]
    l = (labels*255.).astype(np.uint8) # [0,255]

    l = cv2.cvtColor(l, cv2.COLOR_GRAY2RGB)
    l = cv2.LUT(l, build_colormap())  # LUT 是一种映射表，通过查表可以实现对图像像素值的变换操作。

    # SWT中像素值越大代表越宽咯
    swt = (255*swt/swt.max()).astype(np.uint8)   # 将swt的值范围映射到 0 到 255，以便将其保存为图像文件
    cv2.imshow('swt_before_filter', swt)
    # 计算除了0之外的所有像素的中位数
    # median = np.median(swt[swt > 0])
    # TODO TODO TODO
    # 计算上四分位点
    q3 = np.percentile(swt[swt > 0], args.filter_percent)  # 50 means median; default=75
    # 过滤掉分位点以下的值
    swt[swt < q3] = 0

    
    cv2.imshow('Image', im)
    cv2.imshow('Edges', edges)
    # cv2.imshow('X', gradients.x)
    # cv2.imshow('Y', gradients.y)
    # cv2.imshow('Theta', theta)
    cv2.imshow('swt_final', swt)
    # cv2.imshow('Connected Components/labels', l)

    # img = cv2.imread(swt_img)
    gray = swt
    cv2.imshow('gray', gray)
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    # bin： findcontours需要输入二值化图片
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY) # <= 1 -> 0, > 1 -> 255
    cv2.imshow('thresh', thresh)
    # 创建一个膨胀的核（kernel）
    kernel = np.ones((args.dilate_kernel, args.dilate_kernel), np.uint8)  # 核越大 膨胀的越狠  TODO TODO
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    cv2.imshow('dilated_thresh', thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED) # filled会填充内部
    cv2.imshow('mask', mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('Connected_Components.png', l)
    cv2.imwrite('Stroke_Width_Transformed.png', swt)
    cv2.imwrite('mask.png', mask)


if __name__ == '__main__':
    main()

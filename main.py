import random

import numpy
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


class SampleGenerator:
    @staticmethod
    def get_samples():
        source = [
            [   # L
                # 1  2  3  4  5  6
                [1, 1, 0, 0, 0, 0],  # 1
                [1, 1, 0, 0, 0, 0],  # 2
                [1, 1, 0, 0, 0, 0],  # 3
                [1, 1, 0, 0, 0, 0],  # 4
                [1, 1, 0, 0, 0, 1],  # 5
                [1, 1, 1, 1, 1, 1],  # 6
            ],
            [   # U
                # 1  2  3  4  5  6
                [1, 0, 0, 0, 0, 1],  # 1
                [1, 0, 0, 0, 0, 1],  # 2
                [1, 0, 0, 0, 0, 1],  # 3
                [1, 0, 0, 0, 0, 1],  # 4
                [1, 1, 1, 1, 1, 1],  # 5
                [0, 1, 1, 1, 1, 0],  # 6
            ],
            [   # T
                # 1  2  3  4  5  6
                [1, 1, 1, 1, 1, 1],  # 1
                [1, 0, 1, 1, 0, 1],  # 2
                [0, 0, 1, 1, 0, 0],  # 3
                [0, 0, 1, 1, 0, 0],  # 4
                [0, 0, 1, 1, 0, 0],  # 5
                [0, 0, 1, 1, 0, 0],  # 6
            ],
            [   # O
                # 1  2  3  4  5  6
                [1, 1, 1, 1, 1, 1],  # 1
                [1, 1, 0, 0, 1, 1],  # 2
                [1, 1, 0, 0, 1, 1],  # 3
                [1, 1, 0, 0, 1, 1],  # 4
                [1, 1, 1, 1, 1, 1],  # 5
                [1, 1, 1, 1, 1, 1],  # 6
            ],
            [   # K
                # 1  2  3  4  5  6
                [1, 0, 0, 0, 1, 1],  # 1
                [1, 0, 0, 1, 1, 0],  # 2
                [1, 0, 1, 1, 0, 0],  # 3
                [1, 1, 1, 1, 0, 0],  # 4
                [1, 1, 0, 1, 1, 0],  # 5
                [1, 0, 0, 0, 1, 1],  # 6
            ]
        ]

        result = []
        for item in source:
            result.append(numpy.array(item, dtype=int))
        return result


class NoiseGenerator:
    @staticmethod
    def dot_noise(img, prob: float, samples: int = 1):
        if samples < 0:
            raise ValueError
        if samples == 1:
            return numpy.where(numpy.random.rand(*img.shape) < prob, 1 - img, img)
        result = []
        for i in range(samples):
            result.append(numpy.where(numpy.random.rand(*img.shape) < prob, 1 - img, img))
        return result


class RBF:
    w2 = None
    win_count = None

    def __init__(self, source_shape, target_num: int, target_images):
        if type(source_shape) is tuple:
            linear_size = numpy.prod(numpy.array(source_shape), dtype=int).item()
        elif type(source_shape) is int:
            linear_size = source_shape
        else:
            raise ValueError

        if target_num < 2:
            raise ValueError

        if type(target_images) is list:
            target_images = numpy.array(target_images)
        if type(target_images) is numpy.ndarray:
            if len(target_images.shape) < 2:
                raise ValueError
            if len(target_images.shape) > 2:
                if target_images.shape[0] != target_num:
                    raise ValueError
                target_images = numpy.reshape(target_images, newshape=(target_num, -1))
                if target_images.shape[1] != linear_size:
                    raise ValueError
        else:
            raise ValueError

        self.w2 = numpy.reshape(numpy.random.random(size=(target_num, linear_size)), newshape=(target_num, linear_size))
        self.w2 /= numpy.reshape(numpy.linalg.norm(self.w2, axis=1), newshape=(target_num, 1))
        self.win_count = numpy.zeros(shape=target_num, dtype=int)
        pass

    def run(self, sample: numpy.ndarray):
        if len(sample.shape) > 1:
            sample = sample.flatten()
        result = self.w2 @ sample
        return result

    def train_step(self, sample: numpy.ndarray, alpha: float = 0.2):
        err = numpy.linalg.norm(self.w2 - sample, axis=1)
        winner = numpy.argmin(err * self.win_count)
        self.w2[winner] += alpha * (sample - self.w2[winner])
        self.w2[winner] /= numpy.linalg.norm(self.w2[winner])
        self.win_count[winner] += 1
        return err[winner]
    pass


def lab():
    params = [0.16, 40, 0.1, 0.94]
    _labels = [
        f"Dot noise probability [0.16]: ",
        f"Number of test samples [40]: ",
        f"Target training error deviation [0.1]: ",
        f"Dataset train/test data rate [0.94]: "
    ]
    _p_types = [float, int, float, float]

    for i in range(len(params)):
        try:
            temp = _p_types[i](input(_labels[i]))
            params[i] = temp if temp > 0 else params[i]
        except ValueError:
            continue

    dataset = SampleGenerator.get_samples()
    initial_shape = dataset[0].shape
    flattened = []
    for item in dataset:
        flattened.append(numpy.reshape(item, newshape=(-1)))

    image_shape = flattened[0].shape
    network = RBF(image_shape, len(dataset), flattened)

    perturbed_data = []
    for i in range(len(flattened)):
        if params[1] > 1:
            sample_pack = NoiseGenerator.dot_noise(flattened[i], params[0], params[1])
            for item in sample_pack:
                perturbed_data.append(item / numpy.linalg.norm(item))
            # perturbed_data.extend(NoiseGenerator.dot_noise(flattened[i], params[0], params[1]))
        else:
            item = NoiseGenerator.dot_noise(flattened[i], params[0])
            perturbed_data.append(item / numpy.linalg.norm(item))
            # perturbed_data.append(NoiseGenerator.dot_noise(flattened[i], params[0]))

    random.shuffle(perturbed_data)
    dataset_delimiter = int(len(perturbed_data) * params[3])
    train_data = perturbed_data[:dataset_delimiter]
    test_data = perturbed_data[dataset_delimiter:]
    for i in range(len(flattened)):
        test_data.append(flattened[i] / numpy.linalg.norm(flattened[i]))
        pass

    for i in range(int(10e4)):
        curr_err_val = network.train_step(random.choice(train_data), 0.7)
        if numpy.all(curr_err_val < params[2]):
            break
        pass
    else:
        print(f"Training step limit exceeded")

    test_outputs = []
    for item in test_data:
        test_outputs.append(network.run(item))

    plot_content = []
    for i in range(len(test_outputs)):
        scores_content = []
        for j in range(len(dataset)):
            scores_content.append(test_outputs[i][j])
        plot_content.append(tuple((
            numpy.reshape(test_data[i], newshape=initial_shape),
            scores_content
        )))

    figures = []
    for item in plot_content:
        fig = plt.figure()
        gs = GridSpec(2, len(dataset), figure=fig)
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.imshow(item[0])
        ax_main.set_axis_off()
        for i in range(len(dataset)):
            ax_target = fig.add_subplot(gs[1, i])
            ax_target.set_title(f"{item[1][i]:.3f}")
            ax_target.set_axis_off()
        figures.append(fig)
    plt.show()
    pass


if __name__ == '__main__':
    lab()

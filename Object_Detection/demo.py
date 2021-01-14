# coding=utf-8
from depthai_utils import *


class Main(DepthAI):
    def __init__(self, file=None, camera=False):
        super().__init__(file, camera)

    def create_nns(self):
        self.create_nn("models/mobilenet-ssd.blob.sh7cmx7NCE1", "model")

    def start_nns(self):
        self.model_in = self.device.getInputQueue("model_in")
        self.model_nn = self.device.getOutputQueue("model_nn")

    labels = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor"
    ]

    def run_model(self):
        nn_data = run_nn(
            self.model_in, self.model_nn, {"data": to_planar(self.frame, (300, 300))}
        )

        results = to_bbox_result(nn_data)
        # print(results)
        # print('=' * 20)
        # print("model", to_tensor_result(nn_data))
        labels = [int(obj[1]) for obj in results if obj[2] > 0.6]
        self.object_coords = [
            frame_norm(self.frame, *obj[3:7]) for obj in results if obj[2] > 0.6
        ]

        if len(self.object_coords) == 0:
            return False

        self.object_frame = [
            self.frame[
            self.object_coords[i][1]: self.object_coords[i][3],
            self.object_coords[i][0]: self.object_coords[i][2],
            ]
            for i in range(len(self.object_coords))
        ]

        if debug:
            for k,bbox in enumerate(self.object_coords):
                self.put_text(self.labels[labels[k]], bbox[:2]+[10, 10])
                self.draw_bbox(bbox, (10, 245, 10))
        return True

    def parse_fun(self):
        self.run_model()

    def cam_size(self):
        self.first_size = (300, 300)


if __name__ == "__main__":
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera).run()

import cv2
import numpy as np

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    for i, c in enumerate(color):
        image[:, :, i] = np.where(mask == 1,
                                  image[:, :, i] * (1 - alpha) + alpha * c,
                                  image[:, :, i])
    return image

def display_instances(image, boxes, masks, class_ids, class_names, scores):
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    colors = random_colors(N)

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        class_id = class_ids[i]
        label = class_names[class_id]
        score = scores[i] if scores is not None else None

        caption = "{} {:.2f}".format(label, score) if score else label

        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color)

    return image

if __name__ == "__main__":
    import os
    import sys
    import random
    import itertools
    import model as modellib

    ROOT_DIR = os.path.abspath("../")
    COCO_DIR = os.path.join(ROOT_DIR, 'samples/coco')
    sys.path.append(COCO_DIR)

    import coco
    
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.75
    
    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode = "inference", model_dir = MODEL_DIR, config = config
    )
    model.load_weights(MODEL_PATH, by_name=True)

    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

    IMG_DIR = os.path.join(ROOT_DIR, "images/images")
    PRED_DIR = os.path.join(ROOT_DIR, "images/pred_images")
    for file in os.listdir(IMG_DIR):
        image = cv2.imread("{}/{}".format(IMG_DIR, file))
        print("{}/{}".format(IMG_DIR, file))
        print(image.shape)
        results = model.detect([image], verbose=0)
        r = results[0]

        image = display_instances(
            image, r["rois"], r["masks"], r["class_ids"], class_names, r["scores"]
        )
        cv2.imwrite("{}/{}".format(PRED_DIR, file), image)
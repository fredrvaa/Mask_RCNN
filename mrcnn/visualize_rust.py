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
    import utils

    ROOT_DIR = os.path.abspath("../")
    RUST_DIR = os.path.join(ROOT_DIR, 'samples/rust')
    sys.path.append(RUST_DIR)

    import rust
    
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    MODEL_PATH = os.path.join(MODEL_DIR, "rust20191211T1417/mask_rcnn_rust_0030.h5")

    class InferenceConfig(rust.RustConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.3
    
    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode = "inference", model_dir = MODEL_DIR, config = config
    )
    model.load_weights(MODEL_PATH, by_name=True)

    class_names = ["BG", "rust"]

    IMAGES_PATH = os.path.join(ROOT_DIR, "images/images")
    PRED_PATH = os.path.join(ROOT_DIR, "images/pred_images")

    AP = []
    #FOR IMAGES
    for file in os.listdir(IMAGES_PATH):
        image = cv2.imread("{}/{}".format(IMAGES_PATH, file))
        results = model.detect([image], verbose=0)
        r = results[0]

        image = display_instances(
            image, r["rois"], r["masks"], r["class_ids"], class_names, r["scores"]
        )

        AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)

        cv2.imwrite("{}/{}".format(PRED_PATH, file), image)

    print("mAP: ", np.mean(APs))

    #CREATED VIDEO
    # image = cv2.imread("{}/{}".format(RUST_DIR, "HQ_rust.jpg"))
    # height, width = image.shape[:2]
    # x, y, h, w = 0, 0, 1000, 1400
    # while True:
    #     frame = image.copy()[y:y+h, x:x+w]
    #     results = model.detect([frame], verbose=0)
    #     r = results[0]

    #     frame = display_instances(
    #         frame, r["rois"], r["masks"], r["class_ids"], class_names, r["scores"]
    #     )
    #     cv2.imshow("frame", frame)

    #     if x < width - w: x = x + 10
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break
        

    

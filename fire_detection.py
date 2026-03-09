from ultralytics import YOLO
import os
import torch
import time

model_path = 'best.pt'
model = YOLO(model_path)

image_directory = "./images_inpaint_sd/pretrained"

conf_values = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

IOU_THRESHOLD = 0.45

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

def main():
    print(f"Starting evaluation on: {image_directory}")
    print(f"Model: {model_path}")
    print(f"Thresholds: {conf_values}")

    stats = {conf: {'fire_has_detected_count': 0, 'img_avg_conf_sum': 0.0} for conf in conf_values}
    
    total_images = 0
    start_time = time.time()

    min_conf = min(conf_values)
    
    files = [f for f in os.listdir(image_directory) if f.lower().endswith(IMG_EXTENSIONS)]
    total_files_count = len(files)
    
    if total_files_count == 0:
        print("No images found in directory")
        return

    for idx, filename in enumerate(files):
        image_path = os.path.join(image_directory, filename)
        total_images += 1
        
        if idx % 50 == 0:
            print(f"Processing {idx}/{total_files_count}: {filename} ...")

        results = model.predict(image_path, conf=min_conf, iou=IOU_THRESHOLD, verbose=False)
        result = results[0]
        
        boxes = result.boxes
        
        names = result.names
        fire_class_id = None
        for k, v in names.items():
            if v == "Fire":
                fire_class_id = k
                break
        
        if fire_class_id is None and len(names) > 0:
            fire_class_id = list(names.keys())[0]

        if len(boxes) == 0:
            continue

        cls_list = boxes.cls.cpu().tolist()
        conf_list = boxes.conf.cpu().tolist()

        for thresh in conf_values:
            
            valid_fire_scores = []
            for c, s in zip(cls_list, conf_list):
                if int(c) == fire_class_id and s >= thresh:
                    valid_fire_scores.append(s)
            
            if len(valid_fire_scores) > 0:
                stats[thresh]['fire_has_detected_count'] += 1
                
                img_avg_score = sum(valid_fire_scores) / len(valid_fire_scores)
                
                stats[thresh]['img_avg_conf_sum'] += img_avg_score

    end_time = time.time()
    print(f"\nProcessing done in {end_time - start_time:.2f} seconds")
    print(f"Total Images Processed: {total_images}")
    print("\nFinal Summary")
    
    final_results = {}

    for conf in conf_values:
        fire_count = stats[conf]['fire_has_detected_count']
        conf_sum = stats[conf]['img_avg_conf_sum']
        
        fire_ratio = fire_count / total_images if total_images > 0 else 0
        
        avg_fire_conf = conf_sum / total_images if total_images > 0 else 0
        
        final_results[conf] = {
            "fire_ratio": fire_ratio,
            "avg_fire_conf": avg_fire_conf
        }
        
        print(f"conf={conf:.2f} Fire Ratio: {fire_ratio:.4f}, Avg Fire Conf: {avg_fire_conf:.4f}")

    return final_results

if __name__ == "__main__":
    main()
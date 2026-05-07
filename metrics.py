import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

from PIL import Image


# for finding the number of true/false positive and true/false negative pixels
def get_tf_pn(ground_truth, image):
    # get the white values from the black and white images
    wbin_gtruth = ground_truth > 0
    wbin_image = image > 0

    # use the bins to find the true/false positive and true/false negative pixels
    tp_values = np.logical_and(wbin_gtruth, wbin_image) #TP: image pixel is correct and white
    fp_values = np.logical_and(np.logical_not(wbin_gtruth), wbin_image) #FP: image pixel is incorrect and white
    tn_values = np.logical_and(np.logical_not(wbin_gtruth), np.logical_not(wbin_image)) #TN: image pixel is correct and black
    fn_values = np.logical_and(wbin_gtruth, np.logical_not(wbin_image)) #FN: image pixel is incorrect and black

    # sum the values
    tp = np.sum(tp_values)
    fp = np.sum(fp_values)
    tn = np.sum(tn_values)
    fn = np.sum(fn_values)

    return tp, fp, tn, fn


# for calculating metrics from the TP, FP, TN, and FN values
def metrics(tp, fp, tn, fn):    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 2*((precision*recall) / (precision + recall)) if  (precision + recall) > 0 else 0
    pixel_accuracy = (tp + tn) / (tp+ fp + tn + fn) if (tp+ fp + tn + fn) > 0 else 0

    return precision, recall, f1, pixel_accuracy


def display_metrics(ground_truth, image):
    tp, fp, tn, fn = get_tf_pn(ground_truth=ground_truth, image=image)
    precision, recall, f1, pixel_accuracy = metrics(tp, fp, tn, fn)

    # display
    print("----- Metrics -----")
    print(f"Precision: {np.round(precision, 3)}\nRecall: {np.round(recall, 3)}\nF1 Score: {np.round(f1, 3)}\nPixel Accuracy: {np.round(pixel_accuracy, 3)}")

    return precision, recall, f1, pixel_accuracy


def analyze_images(folder_path, precision_list, recall_list, f1_list, accuracy_list, size = -1):
    # 
    for image in os.scandir(folder_path):
        if image.is_file():
            image_path = image.path
            img = cv.imread(image_path)

            # getting the name of the corresponding ground truth image from the image file path
            split1 = image_path.split("\\")
            image_number = split1[1].split("_")[0]
            gtruth_name = image_number + ".png"
            
            if size > 0:
                mask = Image.open("SUT Dataset/1-Segmentation/Ground Truth/" + gtruth_name).convert("RGB").resize((size, size), Image.NEAREST)
                gt = np.array(mask)
            else:
                gt = cv.imread("SUT Dataset/1-Segmentation/Ground Truth/" + gtruth_name)

            precision, recall, f1, pixel_accuracy = display_metrics(gt, img)
            precision_list.append(precision) 
            recall_list.append(recall)
            f1_list.append(f1)
            accuracy_list.append(pixel_accuracy)


def pie_chart(tp, fp, tn, fn):
    labels = ["TP", "FP", "TN", "FN"]
    pn = [tp, fp, tn, fn]

    fig, ax = plt.subplots(figsize=(10, 7))
    wedges, t = plt.pie(pn, labels=labels)

    # legend
    ax.legend(wedges, labels,
          title="Metrics",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

    plt.show()



def display_average_pn(folder_path, size = -1):
    average_tp, average_fp, average_tn, average_fn = ([] for i in range(4))
    for image in os.scandir(folder_path):
        if image.is_file():
            image_path = image.path
            img = cv.imread(image_path)

            # getting the name of the corresponding ground truth image from the image file path
            split1 = image_path.split("\\")
            image_number = split1[1].split("_")[0]
            gtruth_name = image_number + ".png"
            if size > 0:
                mask = Image.open("SUT Dataset/1-Segmentation/Ground Truth/" + gtruth_name).convert("RGB").resize((size, size), Image.NEAREST)
                gt = np.array(mask)
            else:
                gt = cv.imread("SUT Dataset/1-Segmentation/Ground Truth/" + gtruth_name)

            tp, fp, tn, fn = get_tf_pn(ground_truth=gt, image=img)
            average_tp.append(tp)
            average_fp.append(fp)
            average_tn.append(tn)
            average_fn.append(fn)

    # pie chart
    pie_chart(sum(average_tp) / len(average_tp), sum(average_fp) / len(average_fp), sum(average_tn) / len(average_tn), sum(average_fn) / len(average_fn))


def main():
    # lists for storing the metrics for the images
    trad_precision, trad_recall, trad_f1, trad_accuracy = ([] for i in range(4))
    dl_precision, dl_recall, dl_f1, dl_accuracy = ([] for i in range(4))

    metrics = {"traditional precision: " : trad_precision,
               "traditional recall: " : trad_recall,
               "traditional F1 score: ": trad_f1,
               "traditional accuracy: ": trad_accuracy,
               "deep learning precision: " : dl_precision,
               "deep learning recall: " : dl_recall,
               "deep learning F1 score: ": dl_f1,
               "deep learning accuracy: ": dl_accuracy}
    
    # folder paths for method results
    trad_folder = "traditional_outputs/masks"
    dl_folder= "unet_outputs/masks"

    display_average_pn(dl_folder, size=512)

    analyze_images(dl_folder, dl_precision, dl_recall, dl_f1, dl_accuracy, size=512)
    analyze_images(trad_folder, trad_precision, trad_recall, trad_f1, trad_accuracy, size=-1)

    # display metrics
    print("\n\\\\\\\\------ Average Metrics ------////\n")
    for key, value in metrics.items():
        if not value:
            continue
        print(f"Average {key}{round(sum(value) / len(value), 5)}")
        print(f"Lowest {key}{round(min(value), 5)}")
        print(f"Highest {key}{round(max(value), 5)}\n")


    
if __name__ == "__main__":
    main()
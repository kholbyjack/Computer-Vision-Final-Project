import cv2 as cv
import numpy as np

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
    # 
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # 
    f1 = 2*((precision*recall) / (precision + recall))
    pixel_accuracy = (tp + tn) / (tp+ fp + tn + fn)

    return precision, recall, f1, pixel_accuracy

def display_metrics(ground_truth, image):
    tp, fp, tn, fn = get_tf_pn(ground_truth=ground_truth, image=image)
    precision, recall, f1, pixel_accuracy = metrics(tp, fp, tn, fn)

    # display
    print(f"Precision: {np.round(precision, 3)} \n Recall: {np.round(recall, 3)}\n F1 Score: {np.round(f1, 3)}\n Pixel Accuracy: {np.round(pixel_accuracy, 3)}")


def main():
    # temporary testing
    ground = cv.imread("SUT Dataset/1-Segmentation/Ground Truth/1.png")
    also_ground = ground
    not_ground = cv.imread("SUT Dataset/1-Segmentation/Ground Truth/10.png")

    display_metrics(ground, also_ground)
    display_metrics(ground, not_ground)
    


if __name__ == "__main__":
    main()
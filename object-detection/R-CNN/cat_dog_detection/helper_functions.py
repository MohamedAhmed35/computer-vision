import cv2
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from scipy.special import softmax
import tensorflow as tf
import os
import pickle

import xml.etree.ElementTree as ET
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# Parse annotation XML
def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = []
    name_tags = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        name_tag = obj.find("name").text

        # Extract bndbox coordinates for only cat and dogs
        if "dog" in name_tag or "cat" in name_tag:
            x1 = int(bndbox.find("xmin").text)
            y1 = int(bndbox.find("ymin").text)
            x2 = int(bndbox.find("xmax").text)
            y2 = int(bndbox.find("ymax").text)
            boxes.append([x1, y1, x2, y2])
            name_tags.append(name_tag)

    return boxes, name_tags


def intersection_over_union(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA: List or tuple representing the first bounding box [x1, y1, x2, y2].
        boxB: List or tuple representing the second bounding box [x1, y1, x2, y2].

    Returns:
        iou: Intersection over Union value (float) between the two bounding boxes.
    """
    if not isinstance(boxA, (tuple, list)):
        raise TypeError("ground truth box must be a list or a tuple")
    
    if not isinstance(boxB, (tuple, list)):
        raise TypeError("Bound box must be a list or a tuple")
    
    # Coordinates of intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Check if there is an overlap
    if xB <= xA or yB <= yA:
        return 0.0

    # Compute intersection area
    interArea = (xB - xA) * (yB - yA)

    # Compute both bounding box areas
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the IoU value
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # Round the IoU value to two decimal places
    return round(iou, 2)



def selectiveSearch(img, numRects):
    """
    Perform Selective Search to generate region proposals for the given image.

    Args:
        img: Input image (numpy array) must be in BGR format
        numRects: Number of region proposals to return.

    Returns:
        proposed_bboxes: List of bounding boxes for the proposals [x1, y1, x2, y2].
        proposed_images: List of cropped region proposal images (resized to 224x224, RGB).
    """
    # Validate input image
    if img is None or len(img.shape) < 2:
        raise ValueError("Invalid image input for selective search.")
    
    if not isinstance(img, np.ndarray):
        raise TypeError("image must be in numpy format")

    # Create Selective Search Segmentation Object
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()

    # Process the image to get the region proposals  
    rects = ss.process()

    # Initialize lists for proposals
    proposed_bboxes = []
    proposed_images = []

    # Extract region proposals up to the specified number
    for rect in rects[:numRects]:  
        x, y, w, h = rect

        # Extract the proposal image
        proposed_img = img[y:y+h, x:x+w]
        proposed_resized = cv2.resize(proposed_img, (224, 224))  # Resize region to 224x224

        # Convert the image to RGB
        proposed_rgb = cv2.cvtColor(proposed_resized, cv2.COLOR_BGR2RGB)

        # Append the bounding box and region image to respective lists
        proposed_bboxes.append([x, y, x+w, y+h])
        proposed_images.append(proposed_rgb)

    return proposed_bboxes, proposed_images



def get_proposal(image_BGR, gt_bboxes: list, image_counter, num_regions: int, gt_objects, iou_threshold: float, dist_dir):
    """
    Generate labeled proposals for object detection using Selective Search and IoU filtering.

    Args:
        image: Input image (numpy array) in BGR format.
        gt_bboxes: List of ground truth bounding boxes for the objects [x1, y1, x2, y2].
        image_counter: Track the image counts
        num_regions: Number of region proposals to generate.
        gt_box_objects: Corresponding to the object in ground truth bounding boxes.
        iou_threshold: IoU threshold for positive classification.
        dist_dir: The folder where the image will be saved

    Returns:
        image_counter: An int value that tracks the number of saved images
        curr_df: A dataframe that contain information about proposed and ground truth images.
                 ["image_name", "x1", "y1", "x2", "y2", "IOU", "Target"]
    """

    # Validate inputs
    if not isinstance(gt_bboxes, (list, tuple)):
        raise TypeError("gt_bboxes must be a list or tuple")
    if not isinstance(num_regions, int):
        raise TypeError("num_regions must be an int")
    if not isinstance(gt_objects, list):
        raise TypeError("gt_objects must be a list")
    if not os.path.isdir(dist_dir):
        raise ValueError(f"dist_dir at {dist_dir} does not exist.")
    
    
    curr_df = pd.DataFrame(columns=["image_name", "x1", "y1", "x2", "y2", "IOU", "class", "object"])

    # Get proposed regions and their bounding boxes
    bboxes, proposed_regions = selectiveSearch(image_BGR, num_regions)

    # Convert to RGB channel
    image_rgb = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)


    gt_classes = []             # store class correspond to object found in the ground truth box
    gt_bbox_coordinates = []    # Store coordiates of each ground truth box
    
    # Iterate over each ground truth bounding box
    for gt_idx, gt_bbox in enumerate(gt_bboxes):
        gt_bbox_coordinates.append(gt_bbox)

        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox

        # Save the ground truth image
        gt_image = image_rgb[gt_y1:gt_y2, gt_x1: gt_x2]     # Extract the bounding box
        gt_image_resized = cv2.resize(gt_image, (224, 224))
        gt_image_name = f"gt_{image_counter}.png"
        gt_image_path = os.path.join(dist_dir, gt_image_name)
        mpimg.imsave(gt_image_path, gt_image_resized)
        
        # Set the class based on the object type in the image
        if "cat" in gt_objects[gt_idx].lower():       # objects in the image are cats
            gt_class = 0   # Class 0 for cat
        elif "dog" in gt_objects[gt_idx].lower():     # objects in the image are dogs
            gt_class = 1    # Class 1 for dog
        else:
            raise ValueError(f"Unexpected name '{gt_objects}' found. Expected 'cat' or 'dog'.")
        
        gt_classes.append(gt_class)

        # Save ground truth bounding box details: ["image_name", "x1", "y1", "x2", "y2", "class", "object"]
        curr_df.loc[image_counter] = [gt_image_name, gt_x1, gt_y1, gt_x2, gt_y2, 1, gt_class, gt_objects[gt_idx]]

        image_counter += 1    # Increament after saving the ground truth image details


    # Iterate over each proposed bounding box
    for bbox_idx, bbox in enumerate(bboxes):
        # Store the values of IOU
        ious = []   

        # Calculate the IOU value based on each gt_bbox
        for gt_bbox_idx, gt_class in enumerate(gt_classes):
            # Compare ground truth box with region box using IoU
            iou = intersection_over_union(gt_bbox_coordinates[gt_bbox_idx], bbox)
            ious.append(iou)

        # get the maximum IOU value corresponding to max region intersection with ground truth box        
        iou = max(ious)        
        max_iou_idx = ious.index(iou)   # Get the index of max IOU value
        object_class = gt_classes[max_iou_idx]  # Get the class corresponds to max IOU value
        object_name = gt_objects[max_iou_idx]   # choose the object based on max IOU value

        # Determine class label based on IoU
        if iou < iou_threshold:
            object_class = 2    # class 2 for background [region proposals IOU less than the threshold]
            object_name = "background"

        # Save the proposed image
        bbox_name = f"{image_counter}.png"
        bbox_path = os.path.join(dist_dir, bbox_name)
        mpimg.imsave(bbox_path, proposed_regions[bbox_idx])

        x1, y1, x2, y2 = bbox   # Coordinates of proposed bounding box
        # Save proposed bounding box details: ["image_name", "x1", "y1", "x2", "y2", "class", "object"] 
        curr_df.loc[image_counter] = [bbox_name, x1, y1, x2, y2, iou, object_class, object_name]

        image_counter += 1    # Increment after saving the proposed image details 

    return image_counter, curr_df



def process_images_and_annotations(src_dir, dest_dir, annot_dir, image_labels, num_regions = 50, iou_threshold = 0.5):
    """ 
    Generate the proposal images and get the annotation to ground truth box then return the proposed name and proposed image details in dataframe

    Args:
        src_dir: source directory of images
        dest_dir: destination direction where the generated images will be saved
        annot_dir: annotation directory that contains xml files
        image_labels: a List of image labels
        num_regions: An int to set the maximum number of proposed regions
        iou_threshold: A float to determine the class in get_proposal function

    Returns:
        df: A dataframe that contains all information about the proposed and ground truth images for "image_labels"
    """
    if not os.path.isdir(src_dir):
        raise ValueError(f"The source directory at {src_dir} does not exist.")
    if not os.path.isdir(dest_dir):
        raise ValueError(f"The destination directory at {dest_dir} does not exist.")
    if not os.path.isdir(annot_dir):
        raise ValueError(f"Annotation directory at {annot_dir} does not exist.")
    if not isinstance(image_labels, list):
        raise TypeError(f"image_labels must be a list")
    if not isinstance(num_regions, int):
        raise TypeError(f"num_regions must be int")
    if not isinstance(iou_threshold, float):
        raise TypeError(f"iou_threshold must be float")
    
    image_counter = 0   # Track the total number of images 

    # Initialize lists for proposal images and their labels
    df = pd.DataFrame(columns=["image_name", "x1", "y1", "x2", "y2", "IOU", "class", "object"])

    for label in image_labels:  
        
        image_name = label + ".jpg"  
        img_path = os.path.join(src_dir, image_name)

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"The file {img_path} does not exist or is not a file.")  
        
        img = cv2.imread(img_path)  
        
        # Check if image was loaded successfully  
        if img is not None:  
            img_copy = np.copy(img)  
            
            # Parse annotation file for bounding boxes  
            annot_name = label + ".xml"  
            annot_path = os.path.join(annot_dir, annot_name)  
            gt_boxes, name_tags = parse_annotation(annot_path)  
            image_counter, curr_df = get_proposal(img_copy, gt_boxes, image_counter, num_regions, name_tags, iou_threshold, dest_dir)
            
            df = pd.concat([df, curr_df], axis = 0)

    print(f"All image labels are loaded successfully") 
    
    return df


    # Extrat feature of an image
def extract_features(img_array, model):
    # Check inputs
    if not isinstance(img_array, np.ndarray):
        raise TypeError("image must be in numpy format")
    
    if img_array is not None:
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis = 0)     # Add batch dimension

        features = model.predict(img_array, verbose = 0)
    
    return features.flatten()


# Feature Extraction on the Dataset
def extract_features_from_folder(folder_path, dataframe, object_type: str, model):
    """
    loads image from "folder_path" based on some conditions then get its feature vecter
    
    Args:
        folder_path: A directory where the images are stored.
        dataframe: A csv file that contains information about images. ["image_name", "x1", "y1", "x2", "y2", "class", "object"]
        object_type: A string that determine the object we want the SVM later to classify
        model: A pre-trained CNN model that will be used to extract features from an image.
    """
    # Check inputs
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"folder_path at {folder_path} does not exist!")
    if dataframe is None or not isinstance(dataframe, pd.DataFrame):
        raise ValueError(f"A valid pandas DataFrame must be provided.")
    if not isinstance(object_type, str):
        raise TypeError(f"object_type must be a str.")

    features = []
    labels = []

    for _, row in dataframe.iterrows():
        IOU = row["IOU"]
        target = row["object"]
        image_label = row["image_name"]
        image_path = os.path.join(folder_path, image_label)

        # Check the path of the image if exists
        if not os.path.exists(image_path):
            raise ValueError(f"image at {image_path} is not exists")
    
        # Resize image to model's input size
        img = load_img(image_path, target_size = (224, 224), color_mode='rgb')
        img_array = img_to_array(img)                       # Convert to array

        # Set region proposal that its IOU <= 0.3 negative class
        if IOU <= 0.3:
            feature_vector = extract_features(img_array, model)
            features.append(feature_vector)
            labels.append(1)
        # Choose the ground truth region
        elif 1 == IOU and target == object_type:
            if target == object_type:
                # Set only the ground truth of object_type as positive class
                feature_vector = extract_features(img_array, model)
                features.append(feature_vector)
                labels.append(0)
        elif target != object_type and 0.5 <= IOU:
            feature_vector = extract_features(img_array, model)
            features.append(feature_vector)
            labels.append(1)

        
    print(f"{object_type} images are extracted successfully")

    return np.array(features), np.array(labels)



def calc_transformation_params(gt_bbox, bbox):
    """
    Calculate the transformation paramter tx, ty, tw, th based on gt_bbox and bbox coordinates

    Args:
        gt_bbox: coordinates of ground truth bounding box [gt_x1, gt_y1, gt_x2, gt_y2]
        bbox: coordiantes of region proposal bounding box [x1, y1, x2, y2] 
    Returns:  
    Tuple of transformation parameters (tx, ty, tw, th) where:  
    - tx: translation in x direction  
    - ty: translation in y direction  
    - tw: width transformation  
    - th: height transformation

    Raises:  
        TypeError: If gt_bbox is not a list.  
        ValueError: If gt_bbox does not contain exactly four elements.  
        AssertionError: If ground truth coordinates are not valid.  
    """

    # Inputs check
    if not isinstance(gt_bbox, list):
        raise TypeError("gt_bbox must be a list")
    if not isinstance(bbox, list):  
        raise TypeError("bbox must be a list")
    if len(gt_bbox) != 4:
        raise ValueError("gt_bbox length must contain four parameters")
    if len(bbox) != 4:  
        raise ValueError("bbox length must contain four parameters")  


    gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox  
    x1, y1, x2, y2 = bbox  

    # Validate ground truth bounding box coordinates  
    assert gt_x1 < gt_x2, "Ground truth x1 must be less than x2."  
    assert gt_y1 < gt_y2, "Ground truth y1 must be less than y2." 
    # Validate proposed bound box coordinaes 
    assert x1 < x2, "Proposed bounding box x1 must be less than x2."  
    assert y1 < y2, "Proposed bounding box y1 must be less than y2."

    # Calculate width and height for both bounding boxes
    gt_width = gt_x2 - gt_x1
    gt_height = gt_y2 - gt_y1
    width = x2 - x1
    height = y2 - y1

    # Calculatet the center point relative to the origin of the whole image for both bounding boxes 
    # Ground Truth Center
    Gx = gt_width / 2 + gt_x1
    Gy = gt_height/ 2 + gt_y1
    # Proposed Box Center
    Px = width / 2 + x1
    Py = height / 2 + y1

    # Calculate translation parameters (tx, ty)
    tx = (Gx - Px)/ width
    ty = (Gy - Py)/ height

    # Calculate width and height transformation parameters (tw, th)
    tw = np.log(gt_width/width)
    th = np.log(gt_height/height)

    return tx, ty, tw, th


def get_regressor_data(folder_path, dataframe):
    """Extracts proposed images and transformation parameters from a given DataFrame.  

    Args:  
        folder_path (str): The path to the directory containing images.  
        dataframe (pd.DataFrame): A DataFrame with image data and IOU values. 

    Raises:  
        FileNotFoundError: If the specified folder_path does not exist.  
        TypeError: If the provided dataframe is not a pandas DataFrame.  
        ValueError: If the provided DataFrame is empty.  

    Returns:  
        tuple: A tuple containing two numpy arrays:  
            - all_proposed_images: Numpy array of processed image arrays.  
            - all_transform_params: Numpy array of transformation parameter lists.  
    """ 

    # Check if the folder_path exists
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"folder_path at {folder_path} does not exist.")
    # Ensure the provided argument is a valid pandas DataFrame
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("A valid pandas DataFrame must be provided")
    # Continue processing the DataFrame  
    if dataframe.empty:  
        raise ValueError("The DataFrame is empty.")
    
    # Lists to store proposed images and transformation parameters
    all_proposed_images = []
    all_transform_params = []
    all_gt_bbox = []

    # Iterate over each row of the DataFrame
    for _, row in dataframe.iterrows():
        IOU = row["IOU"]
        
        if IOU == 1:
            gt_bbox = row[["x1", "y1", "x2", "y2"]].values.tolist()
            all_gt_bbox.append(gt_bbox)

        # Process rows where IOU is between 0.6 and 1
        if 1 > IOU > 0.6:
            proposed_image = row["image_name"]
            image_path = os.path.join(folder_path, proposed_image)

            # Load and preprocess the image
            img = load_img(image_path, target_size = (224, 224), color_mode='rgb')
            if img:
                img_array = img_to_array(img)/255.0           # Convert to array and normalize it
                all_proposed_images.append(img_array)
            else:
                continue

            bbox = row[["x1", "y1", "x2", "y2"]].values.tolist()

            curr_gt_bbox = None
            # Find the ground truth bounding box and calculate IOU
            for curr_gt_bbox in all_gt_bbox[::-1]:  # Reversing the list because the last appended "gt_bbox" is the nearest to "bbox"
                
                matched_iou = intersection_over_union(curr_gt_bbox, bbox)
                
                # Break the loop if the matched IOU is equal to the current IOU
                if IOU == matched_iou:
                    break

            # Calculate transformation parameters and append to the list
            transform_params = calc_transformation_params(curr_gt_bbox, bbox)
            transform_params = np.array(transform_params)
            all_transform_params.append(transform_params)

    # Return the results as numpy arrays  
    return np.array(all_proposed_images), np.array(all_transform_params)



def non_maximum_suppression(boxes, scores, iou_threshold):
    """   
    Applies Non-Maximum Suppression (NMS) to suppress overlapping bounding boxes.  

    Args:  
        boxes (list): List of bounding boxes, where each box is represented as an array-like object (e.g., [x1, y1, x2, y2]).  
        scores (list): List of scores associated with each bounding box, indicating the confidence level of detection.  
        iou_threshold (float): Intersection over Union (IoU) threshold for determining whether boxes overlap significantly.  

    Returns:  
        selected_indices (list): List of indices of the bounding boxes that are retained after applying NMS.  
    """  

    # Input checks  
    if not isinstance(boxes, list) or len(boxes) == 0:  
        raise ValueError("Expected 'boxes' to be a non-empty list.")  

    if not all(isinstance(box, (tuple, list, np.ndarray)) and len(box) == 4 for box in boxes):  
        raise ValueError("Each box in 'boxes' should be an array-like object with length 4 (x1, y1, x2, y2).")  
        
    if not isinstance(scores, list) or len(scores) != len(boxes):  
        raise ValueError("Expected 'scores' to be a list of the same length as 'boxes'.")  

    if not all(isinstance(score, (int, float)) for score in scores):  
        raise ValueError("All elements in 'scores' should be numeric (int or float).")  
        
    if not isinstance(iou_threshold, (int, float)) or not (0 <= iou_threshold <= 1):  
        raise ValueError("Expected 'iou_threshold' to be a float between 0 and 1.") 

    # Sort the indices of the bounding boxes by their scores in descending order
    sorted_indices = np.argsort(scores)[::-1]

    # Initialize a list to keep track of selected indices of boxes
    selected_indices = []

    # Continue processing while there are still indices to consider
    while len(sorted_indices) > 0:
        # Select the index of the box with the highest score
        current_index = sorted_indices[0]
        selected_indices.append(current_index)  # Add this index to the selected list

        # Remaining indices are those not yet selected 
        remaining_indices = sorted_indices[1:]
        to_keep = []    # List to keep track of indices that will be retained

        # Iterate over the remaining indices to check for IoU
        for i in remaining_indices:
            iou = intersection_over_union(boxes[current_index], boxes[i])

            # If the IoU is less than or equal to the threshold, keep this box
            if iou <= iou_threshold:
                to_keep.append(i)
                
        # Update sorted_indices to only include the boxes that are not suppressed
        sorted_indices = to_keep

    # Return the list of selected indices after applying NMS
    return selected_indices


def load_models():
    # Load the trained models
    classification_model = tf.keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/R-CNN/models/classification.keras")
    regressor_model = tf.keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/R-CNN/models/regressor.keras")

    # Load the model using pickle  
    with open('/content/drive/MyDrive/Colab Notebooks/R-CNN/models/cat_svm_model.pkl', 'rb') as file:  
        cat_svm_model = pickle.load(file) 

    # Load the model using pickle  
    with open('/content/drive/MyDrive/Colab Notebooks/R-CNN/models/dog_svm_model.pkl', 'rb') as file:  
        dog_svm_model = pickle.load(file)

    return classification_model, cat_svm_model, dog_svm_model, regressor_model



def process_image_regions(regions, pred_bboxes, img):  
    """   
    Processes image regions to extract features and classify them using pre-trained models.  
    
    Args:  
        regions: List of regions proposed by the selective search.  
        pred_bboxes: List of predicted bounding boxes corresponding to each region.  
        img: The original image used for extracting features.  

    Returns:  
        boxes (list): List of bounding boxes for detected objects.  
        scores (list): List of scores representing the confidence of the predictions.  
        class_labels (list): List of predicted class labels corresponding to the bounding boxes.  
        adjustment_params (list): List of parameters used to adjust the bounding box predictions.  
    """  

    # Input checks  
    if not isinstance(regions, list) or not all(isinstance(region, (np.ndarray, list)) for region in regions):  
        raise TypeError("Expected 'regions' to be a list of arrays or lists representing regions.")  
    
    if not isinstance(pred_bboxes, list) or not all(isinstance(bbox, (tuple, list, np.ndarray)) and len(bbox) == 4 for bbox in pred_bboxes):  
        raise TypeError("Expected 'pred_bboxes' to be a list of bounding boxes, each represented by an array-like object of shape (4,).")  
    
    if not isinstance(img, np.ndarray) or img.ndim != 3:  
        raise TypeError("Expected 'img' to be a 3D numpy array (height, width, channels).")

    # Load the pre-trained classification models and regressor model
    cls_model, cat_model, dog_model, regressor_model = load_models()

    # Initialize lists to store results
    boxes = []  
    scores = []  
    class_labels = []  
    adjustment_params = []  

    for idx, region in enumerate(regions):  
        curr_bbox = pred_bboxes[idx]  
        
        # Extract features from the current region using the classifier model
        features = extract_features(region, cls_model)

        if len(features.shape) == 1:
            # Add a batch dimension to the features for model input
            features = np.expand_dims(features, 0)  # Add batch dimension  

        # cat_pred = cat_model.predict(features)  
        # dog_pred = dog_model.predict(features)  

        # Get decision scores for cat and dog classifiers (using decision function)
        cat_score = cat_model.decision_function(features)[0]  
        dog_score = dog_model.decision_function(features)[0]  

        # Check if both scores indicate background; if so, skip this region 
        if cat_score > 0 and dog_score > 0:   # means both predictions are Noise
            continue  

        # Combine scores for softmax normalization; using negative scores to adjust for classification   
        combined_scores = np.array([-dog_score, -cat_score])    # Adjusting scores for softmax; the score for class 0 in SVM is a negative and class 1 is positive
                                                                # by multiplying with negative we ensure that the class 0 has a positive value (high probability)
                                                                # while class 1 has a negative value (low probability)
        # Normalize scores to probabilities                                                         
        probabilities = softmax(combined_scores)  

         # Get the highest score and corresponding predicted class
        score = probabilities[np.argmax(probabilities)]       
        classes = ['Dog', 'Cat']    # class labels for decoding
        predicted_class = classes[np.argmax(probabilities)]  

        # Prepare the image for regression model input
        if len(img.shape) == 3:
            img_reshaped = np.expand_dims(img, 0)   # Add batch dimension
        else:
            img_reshaped = img 
    
        # Check if all pixel values in the reshaped image are within the range [0, 1]
        if np.max(img_reshaped) <= 1:
            img_rescaled = img_reshaped
        else:
            # Normalize the pixel values to the range [0, 1] by dividing by 255.0
            img_rescaled = img_reshaped / 255.0  

        # Predict transformation parameters to adjust the bounding box
        transformation_params = regressor_model.predict(img_rescaled, verbose = 0)  

        # Store results for the current region
        scores.append(score)  
        class_labels.append(predicted_class)  
        boxes.append(curr_bbox)  
        adjustment_params.extend(transformation_params)    # Extend to include all transformation params
    
    
    # Return collected results
    return boxes, scores, class_labels, adjustment_params 


def adjust_bounding_box(curr_bbox, params):  
   """  
   Adjusts the bounding box coordinates based on the provided adjustment parameters.  

   Parameters:  
   - curr_bbox: A tuple or list containing the original bounding box coordinates (x1, y1, x2, y2).  
   - params: A tuple or list containing the adjustment parameters (tx_prime, ty_prime, tw_prime, th_prime).  

   Returns:  
   - A tuple containing the adjusted bounding box coordinates (x1_prime, y1_prime, x2_prime, y2_prime).  
   """

   # Validate 'curr_bbox'
   if not isinstance(curr_bbox, (list, tuple)):
      raise TypeError("Expected 'curr_bbox' to be a list or tuple.")
   if len(curr_bbox) != 4:
      raise ValueError("Expected 'curr_bbox' to contain exactly 4 elements: (x1, y1, x2, y2).")
   
   # Validate 'params'
   if not isinstance(params, (list, tuple, np.ndarray)):
      raise TypeError("Expected 'params' to be a list or tuple.")
   if len(params) != 4:
      raise ValueError("Expected 'params' to contain exactly 4 elements: (tx_prime, ty_prime, tw_prime, th_prime).")

   # Unpack the inputs
   x1, y1, x2, y2 = curr_bbox  
   tx_prime, ty_prime, tw_prime, th_prime = params  

   # Original width and height  
   width = x2 - x1  
   height = y2 - y1  
   
   # Compute the center of the predicted box  
   Px = (x1 + x2) / 2  
   Py = (y1 + y2) / 2  

   # Get the corrected center  
   Gx_prime = width * tx_prime + Px  
   Gy_prime = height * ty_prime + Py  
   
   # Compute the corrected width and height  
   w_prime = width * np.exp(tw_prime)  
   h_prime = height * np.exp(th_prime)  

   # Compute corrected coordinates  
   x1_prime = int(Gx_prime - w_prime / 2)  
   y1_prime = int(Gy_prime - h_prime / 2)  
   x2_prime = int(Gx_prime + w_prime / 2)  
   y2_prime = int(Gy_prime + h_prime / 2)  

   return x1_prime, y1_prime, x2_prime, y2_prime
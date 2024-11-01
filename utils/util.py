import string, csv, cv2
import numpy as np


# alphabet = {A, B, E, K, M, H, O, P, C, T, Y, X}
# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'A': '4',
                    'B': '8',
                    'T': '7',
                    'C': '6'}

dict_int_to_char = {value: key for key, value in dict_char_to_int.items()}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
             'license_number_score'])

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id] and 'license_plate' in results[frame_nmr][car_id] and 'text' in \
                        results[frame_nmr][car_id]['license_plate']:
                    car_bbox = str(results[frame_nmr][car_id]['car']['bbox']).replace(',', '')
                    license_plate_bbox = str(results[frame_nmr][car_id]['license_plate']['bbox']).replace(',', '')
                    license_plate_text = results[frame_nmr][car_id]['license_plate']['text']
                    license_plate_bbox_score = results[frame_nmr][car_id]['license_plate']['bbox_score']
                    license_plate_text_score = results[frame_nmr][car_id]['license_plate']['text_score']

                    writer.writerow(
                        [frame_nmr, car_id, car_bbox, license_plate_bbox, license_plate_bbox_score, license_plate_text,
                         license_plate_text_score])




def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 8 and len(text) != 9:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()) and \
       (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in dict_char_to_int.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char,
               1: dict_char_to_int,
               2: dict_char_to_int,
               3: dict_char_to_int,
               4: dict_int_to_char,
               5: dict_int_to_char,
               6: dict_char_to_int,
               7: dict_char_to_int,
               8: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6, 7]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    if len(text) == 9:
        if text[8] in mapping[8].keys():
            license_plate_ += mapping[8][text[8]]
        else:
            license_plate_ += text[8]
    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        # if license_complies_format(text):
            # return format_license(text), score
        return text, score

    return 0, 0


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1

def draw_rectangle_1_percentile(image):
    target_pixels = int(image.shape[0] * image.shape[1] * 0.01)

    check_color = (0, 255, 0)
    square_size = int(np.sqrt(target_pixels))
    rectangle_width = int(np.sqrt(target_pixels * 4.642857142857143))
    rectangle_height = int(np.sqrt(target_pixels / 4.642857142857143))
    square_width = int(np.sqrt(target_pixels * 1.705882353))
    square_height = int(np.sqrt(target_pixels / 1.705882353))

    cv2.rectangle(image, (0, 0), (square_width, square_height), check_color, 1)  
    cv2.rectangle(image, (0, 0), (rectangle_width, rectangle_height), check_color, 1)  
    cv2.putText(image, '1%', (10, round(image.shape[0]/25)), cv2.FONT_HERSHEY_PLAIN, (image.shape[0]/300), check_color, 1)
    return image

def get_crop_from_frame(frame, crop):
#     print('==========', frame.shape, len(crop))
    crop = list(map(round, crop))
    result = frame[crop[1]:crop[3],crop[0]:crop[2]].copy()
    return result

def fix_broken_ids(json_data, time_ns, car_id, data):
    """
    Replace car id based on previous text detections. 
    
    Arguments
    ---------
    json_data: dict
        dict with all detections
    time_ns: int 
        timestamp for current frame
    car_id: int
        id of candidate for replacement
    data: dict
        information about numberplate detection for current car_id
    
    Returns
    -------
    json_data: dict
        dict with all detections where car ids fixed
    car_id: int
        replaced id
    car_id_2: int
        new id
    """
    car_id = int(car_id)
    try:
        timestamps = list(json_data.keys())
        if len(timestamps) > 1:
            for item in timestamps[-2::-1]:
                for car_id_2 in json_data[item]:
                    if car_id_2 != "frame" and json_data[item][car_id_2]["pred_text"] == data["pred_text"] and data["pred_text"] != "unreadable" and car_id != car_id_2:
                        print(f"Id replace {car_id} = {car_id_2}")
                        json_data[time_ns].pop(car_id, None)
                        json_data[time_ns][car_id_2] = data
                        return json_data, car_id, car_id_2
    except KeyError:
        pass

if __name__ == "__main__":
    print('True', license_complies_format('A888AA88'))
    print('True', license_complies_format('A888AA123'))
    print('True', license_complies_format('0ABO48ABO'))
    print('True', license_complies_format('A8T8AA88'))
    print('False', license_complies_format('88AA88'))
    print('False', license_complies_format('3888AA88'))
    
    print('0ABO48ABO =>', format_license('0ABO48ABO'))

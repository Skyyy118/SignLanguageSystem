def extract_landmarks(hand_landmarks):

    landmark_list = []

    # extract coordinates
    for lm in hand_landmarks.landmark:
        landmark_list.append([lm.x, lm.y, lm.z])

    if len(landmark_list) == 0:
        return None

    # wrist as origin
    base_x, base_y, base_z = landmark_list[0]

    relative_landmarks = []

    for x, y, z in landmark_list:
        relative_landmarks.append(x - base_x)
        relative_landmarks.append(y - base_y)
        relative_landmarks.append(z - base_z)

    # normalize
    max_value = max(map(abs, relative_landmarks))

    if max_value == 0:
        max_value = 1

    normalized_landmarks = [n / max_value for n in relative_landmarks]

    return normalized_landmarks
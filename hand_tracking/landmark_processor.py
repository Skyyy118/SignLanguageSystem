def extract_landmarks(hand_landmarks):
    landmark_list = []

    # Step 1: Extract raw coordinates
    for lm in hand_landmarks.landmark:
        landmark_list.append([lm.x, lm.y, lm.z])

    # Step 2: Convert to relative coordinates (subtract wrist)
    base_x, base_y, base_z = landmark_list[0]

    relative_landmarks = []
    for x, y, z in landmark_list:
        relative_landmarks.extend([
            x - base_x,
            y - base_y,
            z - base_z
        ])

    # Step 3: Normalize (scale)
    max_value = max(abs(val) for val in relative_landmarks)

    normalized_landmarks = [
        val / max_value for val in relative_landmarks
    ]

    return normalized_landmarks
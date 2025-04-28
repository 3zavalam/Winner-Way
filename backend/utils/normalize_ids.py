def normalize_by_proximity(tracked_frames, target_id=0):
    """
    Asegura que el jugador más cercano a la cámara tenga siempre el `target_id`.
    """
    normalized = []

    for frame in tracked_frames:
        if not frame:
            normalized.append(frame)
            continue

        # Encontrar jugador más cercano (mayor y2)
        max_y = -1
        main_id = None
        for det in frame:
            if det["y2"] > max_y:
                max_y = det["y2"]
                main_id = det["id"]

        new_frame = []
        for det in frame:
            new_det = det.copy()
            if det["id"] == main_id:
                new_det["id"] = target_id
            elif det["id"] == target_id:
                new_det["id"] = main_id  # swap si ya existía
            new_frame.append(new_det)

        normalized.append(new_frame)

    return normalized

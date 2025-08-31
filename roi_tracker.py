import numpy as np

def calculate_iou(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Using Intersection over Minimum Area to better handle containment
    min_area = min(boxAArea, boxBArea)
    if min_area == 0:
        return 0.0

    # Return the ratio of intersection to the smaller of the two areas
    return interArea / float(min_area)

class ROITracker:
    def __init__(self):
        self.next_roi_id = 0
        self.previous_rois = {}  # Store rois by id

    def _classify_new_roi(self, roi, layer_index, config):
        if not config.roi_params.enable_raft_support_handling:
            return "model"
        if layer_index < config.roi_params.raft_layer_count and roi['area'] >= config.roi_params.raft_min_size:
            return "raft"
        if roi['area'] <= config.roi_params.support_max_size and layer_index < config.roi_params.support_max_layer:
            return "support"
        return "model"

    def update_and_classify(self, current_rois_raw, layer_index, config):
        if not self.previous_rois:
            # First frame, classify all ROIs as new
            initial_rois = {}
            for roi in current_rois_raw:
                new_id = self.next_roi_id
                self.next_roi_id += 1
                roi['id'] = new_id
                roi['classification'] = self._classify_new_roi(roi, layer_index, config)
                initial_rois[new_id] = roi
            self.previous_rois = initial_rois
            return list(self.previous_rois.values())

        # Match current ROIs with previous ROIs
        current_indices = list(range(len(current_rois_raw)))
        prev_ids = list(self.previous_rois.keys())

        iou_matrix = np.zeros((len(current_rois_raw), len(prev_ids)))
        for i, current_roi in enumerate(current_rois_raw):
            for j, prev_id in enumerate(prev_ids):
                iou_matrix[i, j] = calculate_iou(current_roi['bbox'], self.previous_rois[prev_id]['bbox'])

        matches = {} # current_idx -> prev_id
        used_prev_ids = set()

        # Find best match for each current_roi, sorted by score
        # Using -1 on axis=None flattens and sorts descending
        sorted_indices = np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape)

        for i, j in zip(*sorted_indices):
            # i = current_roi_idx, j = prev_id_idx
            if iou_matrix[i,j] < 0.5: # Use a higher threshold for IoA/IoMin
                break

            prev_id_to_match = prev_ids[j]
            if i not in matches and prev_id_to_match not in used_prev_ids:
                matches[i] = prev_id_to_match
                used_prev_ids.add(prev_id_to_match)

        new_tracked_rois = {}

        # Process matched ROIs
        for current_idx, prev_id in matches.items():
            current_roi = current_rois_raw[current_idx]
            prev_roi = self.previous_rois[prev_id]

            current_roi['id'] = prev_id

            if prev_roi['classification'] == 'support':
                growth = current_roi['area'] / prev_roi['area'] if prev_roi['area'] > 0 else float('inf')
                if growth > config.roi_params.support_max_growth:
                    current_roi['classification'] = 'model'
                else:
                    current_roi['classification'] = 'support'
            else:
                current_roi['classification'] = prev_roi['classification']

            new_tracked_rois[prev_id] = current_roi

        # Handle new ROIs
        for i in range(len(current_rois_raw)):
            if i not in matches:
                roi = current_rois_raw[i]
                new_id = self.next_roi_id
                self.next_roi_id += 1
                roi['id'] = new_id
                roi['classification'] = self._classify_new_roi(roi, layer_index, config)
                new_tracked_rois[new_id] = roi

        self.previous_rois = new_tracked_rois
        return list(self.previous_rois.values())

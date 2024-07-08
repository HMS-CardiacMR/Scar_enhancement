def dice_score(gt, pred):
    intersection = (gt.flatten(-1)*pred.flatten(-1)).sum()
    denominator = (gt.flatten(-1).sum() + pred.flatten(-1).sum())
    score = 2*intersection/denominator
    return score

def recall_seg(groundtruth_mask, pred_mask):
    bool_pred_mask = (pred_mask > 0).byte()
    bool_groundtruth_mask = (groundtruth_mask > 0).byte()
    intersect = (bool_pred_mask.flatten(-1) & bool_groundtruth_mask.flatten(-1)).sum()
    total_pixel_truth = (groundtruth_mask.flatten(-1)).sum()
    recall = (intersect/total_pixel_truth).mean()
    return recall

def mask_difference(gt, pred):
    diff = gt.flatten(-1).sum() - pred.flatten(-1).sum()
    return diff

def iou_score(gt, pred):
    intersection = (gt * pred).sum()
    union = gt.sum() + pred.sum() - intersection
    score = intersection / union
    return score
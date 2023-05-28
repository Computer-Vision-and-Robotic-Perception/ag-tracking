import torch
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import resize_boxes
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FRCNN_FPN(FasterRCNN):
    def __init__(self, backbone_type='ResNet50FPN', num_classes=2, pretrained=False):
        print(f'use backbone: {backbone_type}')
        if backbone_type=='ResNet34FPN':
            backbone = resnet_fpn_backbone('resnet34', pretrained)
        if backbone_type=='ResNet50FPN':
            backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=None)
        if backbone_type=='ResNet101FPN':
            backbone = resnet_fpn_backbone('resnet101', pretrained)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.roi_heads.nms_thresh = 0.5

    def detect_batches(self, images):
        device = list(self.parameters())[0].device
        images = images.to(device)
        self.original_image_sizes = [img.shape[-2:] for img in images]
        images, _ = self.transform(images, None)
        self.preprocessed_image_sizes = images.image_sizes
        # get backbone feature
        features = self.backbone(images.tensors)  # dict('1-4, pool'), tensors inside
        proposals, _ = self.rpn(images, features) # Meshgrid Warning comes from here, (list: 10, Tensors(1000, 4))1
        detections, _ = self.roi_heads(features, proposals, self.preprocessed_image_sizes)
        # post-processing for re-scaling
        detections = self.transform.postprocess(detections, self.preprocessed_image_sizes, self.original_image_sizes)
        return detections, features, self.preprocessed_image_sizes

    def predict_boxes(self, boxes, features, class_id=1):
        """Regress only single image for tracktor
        """
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)
        boxes = resize_boxes(boxes, self.original_image_sizes[0], self.preprocessed_image_sizes[0])
        proposals = [boxes]
        box_features = self.roi_heads.box_roi_pool(features, proposals, [self.preprocessed_image_sizes[0]])
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)
        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        pred_boxes = pred_boxes[:, class_id].detach()
        pred_scores = pred_scores[:, class_id].detach()
        pred_boxes = resize_boxes(pred_boxes, self.preprocessed_image_sizes[0], self.original_image_sizes[0])
        return pred_boxes, pred_scores

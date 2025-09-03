class PostDetect(nn.Module):
    export = True
    shape = None
    dynamic = False
    iou_thres = 0.65
    conf_thres = 0.25
    topk = 100

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        shape = x[0].shape
        b, res, b_reg_num = shape[0], [], self.reg_max * 4
        for i in range(self.nl):
            res.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(
                0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        x = [i.view(b, self.no, -1) for i in res]
        y = torch.cat(x, 2)
        boxes, scores = y[:, :b_reg_num, ...], y[:, b_reg_num:, ...].sigmoid()
        boxes = boxes.view(b, 4, self.reg_max, -1).permute(0, 1, 3, 2)
        boxes = boxes.softmax(-1) @ torch.arange(self.reg_max).to(boxes)
        boxes0, boxes1 = -boxes[:, :2, ...], boxes[:, 2:, ...]
        boxes = self.anchors.repeat(b, 2, 1) + torch.cat([boxes0, boxes1], 1)
        boxes = boxes * self.strides

        return TRT_NMS.apply(boxes.transpose(1, 2), scores.transpose(1, 2),
                             self.iou_thres, self.conf_thres, self.topk)











class TRT_NMS(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx: Graph,
            boxes: Tensor,
            scores: Tensor,
            iou_threshold: float = 0.65,
            score_threshold: float = 0.25,
            max_output_boxes: int = 100,
            background_class: int = -1,
            box_coding: int = 0,
            plugin_version: str = '1',
            score_activation: int = 0
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_boxes, num_classes = scores.shape
        num_dets = torch.randint(0,
                                 max_output_boxes, (batch_size, 1),
                                 dtype=torch.int32)
        boxes = torch.randn(batch_size, max_output_boxes, 4)
        scores = torch.randn(batch_size, max_output_boxes)
        labels = torch.randint(0,
                               num_classes, (batch_size, max_output_boxes),
                               dtype=torch.int32)

        return num_dets, boxes, scores, labels                             

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Changes were made to this file by Hiroaki Yamagiwa.

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from segment_anything import SamAutomaticMaskGenerator
from segment_anything.modeling import Sam
from segment_anything.utils.amg import (MaskData, area_from_rle,
                                        batched_mask_to_box, box_xyxy_to_xywh,
                                        calculate_stability_score,
                                        coco_encode_rle, generate_crop_boxes,
                                        is_box_near_crop_edge,
                                        mask_to_rle_pytorch, rle_to_mask,
                                        uncrop_masks)
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore


def batched_mask_to_prob(masks: torch.Tensor) -> torch.Tensor:
    """
    For implementation, see the following issue comment:

    "To get the probability map for a mask,
    we simply do element-wise sigmoid over the logits."
    URL: https://github.com/facebookresearch/segment-anything/issues/226

    Args:
        masks: Tensor of shape [B, H, W] representing batch of binary masks.

    Returns:
        Tensor of shape [B, H, W] representing batch of probability maps.
    """
    probs = torch.sigmoid(masks).to(masks.device)
    return probs


def batched_sobel_filter(probs: torch.Tensor, masks: torch.Tensor
                         ) -> torch.Tensor:
    """
    For implementation, see section D.2 of the paper:

    "we apply a Sobel filter to the remaining masks' unthresholded probability
    maps and set values to zero if they do not intersect with the outer 
    boundary pixels of a mask."
    URL: https://arxiv.org/abs/2304.02643

    Args:
        probs: Tensor of shape [B, H, W] representing batch of probability maps.
        masks: Tensor of shape [B, H, W] representing batch of binary masks.

    Returns:
        Tensor of shape [B, H, W] with filtered probability maps.
    """
    # probs: [B, H, W]
    # Add channel dimension to make it [B, 1, H, W]
    probs = probs.unsqueeze(1)

    # sobel_filter: [1, 1, 3, 3]
    sobel_filter_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                                  dtype=torch.float32
                                  ).to(probs.device).unsqueeze(0)
    sobel_filter_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                                  dtype=torch.float32
                                  ).to(probs.device).unsqueeze(0)

    # Apply the Sobel filters
    G_x = F.conv2d(probs, sobel_filter_x, padding=1)
    G_y = F.conv2d(probs, sobel_filter_y, padding=1)

    # Combine the gradients
    probs = torch.sqrt(G_x ** 2 + G_y ** 2)

    # Iterate through each image in the batch
    for i in range(probs.shape[0]):
        # Convert binary mask to float
        mask = masks[i].float()

        G_x = F.conv2d(mask[None, None], sobel_filter_x, padding=1)
        G_y = F.conv2d(mask[None, None], sobel_filter_y, padding=1)
        edge = torch.sqrt(G_x ** 2 + G_y ** 2)
        outer_boundary = (edge > 0).float()

        # Set to zero values that don't touch the mask's outer boundary.
        probs[i, 0] = probs[i, 0] * outer_boundary

        # Set to zero values for image border
        margin = 5
        probs[i, 0, 0:margin, :] = 0
        probs[i, 0, -margin:, :] = 0
        probs[i, 0, :, 0:margin] = 0
        probs[i, 0, :, -margin:] = 0

    # Remove the channel dimension
    probs = probs.squeeze(1)

    return probs


class SamAutomaticMaskAndProbabilityGenerator(SamAutomaticMaskGenerator):
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 16,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        nms_threshold: float = 0.7
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
          nms_threshold (float): The IoU threshold used for non-maximal suppression
        """
        super().__init__(
            model,
            points_per_side,
            points_per_batch,
            pred_iou_thresh,
            stability_score_thresh,
            stability_score_offset,
            box_nms_thresh,
            crop_n_layers,
            crop_nms_thresh,
            crop_overlap_ratio,
            crop_n_points_downscale_factor,
            point_grids,
            min_mask_region_area,
            output_mode,
        )
        self.nms_threshold = nms_threshold

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        # Generate masks
        mask_data = self._generate_masks(image)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
                "prob": mask_data["probs"][idx],
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # DO NOT filter by predicted IoU
        # if self.pred_iou_thresh > 0.0:
        #     keep_mask = data["iou_preds"] > self.pred_iou_thresh
        #     data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        # DO NOT filter by stability score
        # if self.stability_score_thresh > 0.0:
        #     keep_mask = data["stability_score"] >= self.stability_score_thresh
        #     data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["probs"] = batched_mask_to_prob(data["masks"])
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # filter by nms
        if self.nms_threshold > 0.0:
            keep_mask = batched_nms(
                data["boxes"].float(),
                data["iou_preds"],
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.nms_threshold,
            )
            data.filter(keep_mask)

        # apply sobel filter for probability map
        data["probs"] = batched_sobel_filter(data["probs"], data["masks"])

        # set prob to 0 for pixels outside of crop box
        # data["probs"] = batched_crop_probs(data["probs"], data["boxes"])

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

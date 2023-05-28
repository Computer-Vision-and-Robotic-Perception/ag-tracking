import cv2
import torch
import numpy as np
from collections import deque
import torch.nn.functional as F
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import clip_boxes_to_image, nms
from .utils import warp_pos, get_center, get_height, get_width, make_pos


class Tracker:
	"""The main tracking file, here is where magic happens."""
	def __init__(self, detector, tracker_cfg):	
		self.detector = detector
		self.detection_person_thresh = tracker_cfg.DETECTION_PERSON_THRESH
		self.regression_person_thresh = tracker_cfg.REGRESSION_PERSON_THRESH
		self.nms_thresh = tracker_cfg.NMS_THRESH

		self.tracks = []
		self.inactive_tracks = []
		self.track_num = 0
		self.im_index = 0
		self.results = {}

	def reset(self, hard=True):
		self.tracks = []
		self.inactive_tracks = []
		if hard:
			self.track_num = 0
			self.im_index = 0
			self.results = {}

	def tracks_to_inactive(self, tracks):
		self.tracks = [t for t in self.tracks if t not in tracks]
		for t in tracks:
			t.pos = t.last_pos[-1]
		self.inactive_tracks += tracks

	def add(self, new_det_pos, new_det_scores):
		"""Initializes new Track objects and saves them."""
		num_new = new_det_pos.size(0)
		for i in range(num_new):
			self.tracks.append(Track(
				new_det_pos[i].view(1, -1),
                new_det_scores[i],
                self.track_num + i
            ))
		self.track_num += num_new

	def regress_tracks(self, blob):
		"""Regress the position of the tracks and also checks their scores."""
		if len(self.tracks):
			pos, _ = self.get_pos()
			boxes, scores = self.detector.predict_boxes(pos, blob['features'], 1) # class id
			pos = clip_boxes_to_image(boxes, blob['img'].shape[-2:])
			for i in range(len(self.tracks) - 1, -1, -1):
				t = self.tracks[i]
				t.score = scores[i]
				t.pos = pos[i].view(1, -1)
				if scores[i] == 0:
					self.tracks_to_inactive([t])

	def get_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.tracks) == 1:
			pos = self.tracks[0].pos
			score = torch.Tensor([self.tracks[0].score.item()]).cuda()
		elif len(self.tracks) > 1:
			pos = torch.cat([t.pos for t in self.tracks], 0)
			score = torch.Tensor([t.score.item() for t in self.tracks]).cuda()
		else:
			pos = torch.zeros(0).cuda()
			score = torch.zeros(0).cuda()
		return pos, score

	def step(self, blob, detections):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
		for t in self.tracks:
			# add current position to last_pos list. BH 1
			t.last_pos.append(t.pos.clone())

		# filter by class id
		boxes, scores, labels = detections['boxes'], detections['scores'], detections['labels']
		det_pos = boxes[labels == 1]     # class id
		det_scores = scores[labels == 1] # class id                                       # by class id

		# Predict tracks
		if len(self.tracks):
			# regress new bbox locations
			self.regress_tracks(blob)

		# BH: Combined approach to filter using nms
		if det_pos.nelement() > 0 and len(self.tracks) > 0:
			pos, score = self.get_pos()
			poses = torch.cat([pos, det_pos])
			scores = torch.cat([score, det_scores])
			# d: new detections, t: existing tracks
			pos_idxs = list(range(len(self.tracks))) + list(range(det_pos.shape[0]))
			pos_type = len(self.tracks) * ['t'] + det_pos.shape[0] * ['d']
			ious = box_iou(poses, poses)
			nms_keep = nms(poses, scores, self.nms_thresh)
			det_pos_keep = torch.zeros(det_pos.shape[0]).cuda().bool()
			trk_pos_keep = torch.zeros(len(self.tracks)).cuda().bool()
			for idx in nms_keep:
				if pos_type[idx] == 'd': # detection
					replaced = False     # replaced a track 	
					supress = torch.gt(ious[idx], self.nms_thresh)
					indexes, lens = [], []
					for i, sup in enumerate(supress):
						if sup.item() and i != idx.item() and pos_type[i] == 't': # if the detection supressed a track
							replaced = True
							self.tracks[pos_idxs[i]].pos = poses[idx].view(1, -1)
							self.tracks[pos_idxs[i]].score = scores[idx]
							indexes.append(pos_idxs[i])
							lens.append(len(self.tracks[indexes[-1]].last_pos))
					if replaced:
						trk_pos_keep[indexes[lens.index(max(lens))]] = True     							
					if not replaced and scores[idx].item() > self.detection_person_thresh:
						det_pos_keep[pos_idxs[idx]] = True
				else: # existing track
					if scores[idx].item() > self.regression_person_thresh:
						trk_pos_keep[pos_idxs[idx]] = True

			det_pos = det_pos[det_pos_keep]
			det_scores = det_scores[det_pos_keep]
			self.tracks_to_inactive([t for i, t in enumerate(self.tracks) if not trk_pos_keep[i].item()])

		elif det_pos.nelement() > 0:
			# BH: filtering by NMS between raw detections to filter
			keep = nms(det_pos, det_scores, self.nms_thresh)
			det_pos = det_pos[keep]
			det_scores = det_scores[keep]

		# Create new tracks
		if det_pos.nelement() > 0:
			self.add(det_pos, det_scores)

		# Generate Results 
		for t in self.tracks:
			if t.id not in self.results.keys():
				self.results[t.id] = {}

			self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu(), np.array([t.score.cpu()])])

		for t in self.inactive_tracks:
			t.count_inactive += 1

		self.inactive_tracks = [t for t in self.inactive_tracks if t.has_positive_area()]
		self.im_index += 1

	def get_results(self):
		return self.results


class Track(object):
	"""This class contains all necessary for every individual track."""
	def __init__(self, pos, score, track_id, back=5):
		self.id = track_id
		self.pos = pos
		self.score = score
		self.count_inactive = 0
		self.last_pos = deque([pos.clone()], maxlen=back) # TODO: determine this

	def has_positive_area(self):
		# is x2 > x1 and y2 > y1
		target = 15
		return self.pos[0, 2] > self.pos[0, 0] + target and self.pos[0, 3] > self.pos[0, 1] + target

	def reset_last_pos(self):
		self.last_pos.clear()
		self.last_pos.append(self.pos.clone())

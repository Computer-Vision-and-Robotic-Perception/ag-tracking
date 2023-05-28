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
	def __init__(self, detector, reid_network, tracker_cfg):
		self.detector = detector
		self.reid_network = reid_network
		self.detection_person_thresh = tracker_cfg.DETECTION_PERSON_THRESH
		self.regression_person_thresh = tracker_cfg.REGRESSION_PERSON_THRESH
		self.nms_thresh = tracker_cfg.NMS_THRESH

		self.do_reid = tracker_cfg.DO_REID
		self.inactive_patience = tracker_cfg.INACTIVE_PATIENCE
		self.max_features_num = tracker_cfg.MAX_FEATURES_NUM
		self.reid_sim_threshold = tracker_cfg.REID_SIM_THRESHOLD

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

	def add(self, new_det_pos, new_det_scores, new_det_features, hw):
		"""Initializes new Track objects and saves them."""
		num_new = new_det_pos.size(0)
		for i in range(num_new):
			track_num = self.track_num + 1
			track = Track(new_det_pos[i].view(1, -1), new_det_scores[i], track_num, 
			              new_det_features[i].view(1, -1), self.max_features_num, hw)
			if track.has_positive_area() and track.isin_edge() or self.im_index == 0:
				self.tracks.append(track)
				self.track_num = track_num

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

	def reid(self, blob, new_det_pos, new_det_scores):
		"""Tries to ReID inactive tracks with provided detections."""
		new_det_features = [torch.zeros(0).cuda() for _ in range(len(new_det_pos))]

		if self.do_reid:
			new_det_features = self.reid_network.test_rois(blob['img'], new_det_pos).data
			if len(self.inactive_tracks) >= 1:
				# calculate appearance distances
				dist_mat, pos = [], []
				for t in self.inactive_tracks:
					dist_mat.append(torch.cat([t.test_features(feat.view(1, -1)) for feat in new_det_features], dim=1))
				if len(dist_mat) > 1:
					dist_mat = torch.cat(dist_mat, 0)
				else:
					dist_mat = dist_mat[0]

				dist_mat = dist_mat.cpu().numpy()
				row_ind, col_ind = linear_sum_assignment(dist_mat)
				assigned = []
				remove_inactive = []
				for r, c in zip(row_ind, col_ind):
					if dist_mat[r, c] <= self.reid_sim_threshold:
						t = self.inactive_tracks[r]
						self.tracks.append(t)
						t.count_inactive = 0
						t.pos = new_det_pos[c].view(1, -1)
						t.reset_last_pos()
						t.add_features(new_det_features[c].view(1, -1))
						assigned.append(c)
						remove_inactive.append(t)
				
				# # BH DEBUG: collecting ReID distances
				# with open('output/reids/dists.txt', 'a') as logger:
				# 	dists = dist_mat.flatten()
				# 	dists = dists[dists < 999]
				# 	if len(dists): logger.write('\n'.join(['%6.3f' % num for num in dists])+ '\n')
				# # BH DEBUG END

				for t in remove_inactive:
					self.inactive_tracks.remove(t)

				keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
				if keep.nelement() > 0:
					new_det_pos = new_det_pos[keep]
					new_det_scores = new_det_scores[keep]
					new_det_features = new_det_features[keep]
				else:
					new_det_pos = torch.zeros(0).cuda()
					new_det_scores = torch.zeros(0).cuda()
					new_det_features = torch.zeros(0).cuda()

		return new_det_pos, new_det_scores, new_det_features

	def add_features(self, new_features):
		"""Adds new appearance features to active tracks."""
		for t, f in zip(self.tracks, new_features):
			t.add_features(f.view(1, -1))

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

		# Combined approach to filter using nms
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
							self.tracks[pos_idxs[i]].pos = poses[idx].view(1, -1) # update the track with the detection bounding box
							self.tracks[pos_idxs[i]].score = scores[idx]          # and score
							indexes.append(pos_idxs[i])                           # collect indexes
							lens.append(len(self.tracks[indexes[-1]].last_pos))   # and lengths
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

			# For ReID
			if len(self.tracks) > 0 and self.do_reid:
				pos, _ = self.get_pos()
				new_features = self.reid_network.test_rois(blob['img'], pos).data
				self.add_features(new_features)
	
		elif det_pos.nelement() > 0:
			# BH: filtering by NMS between raw detections to filter
			keep = nms(det_pos, det_scores, self.nms_thresh)
			det_pos = det_pos[keep]
			det_scores = det_scores[keep]

		# ReID and/or create new tracks
		if det_pos.nelement() > 0:
			new_det_pos = det_pos
			new_det_scores = det_scores

			# try to reidentify tracks, BH 5
			new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

			# add new: use global track increment for multi-cameras: same id in different camera is not allowed
			# add new track starting time, BH 6
			if new_det_pos.nelement() > 0:
				self.add(new_det_pos, new_det_scores, new_det_features, (blob['img'].shape[2], blob['img'].shape[3]))

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

def isin_edge(box, hw):
	edge = False
	if   box[0, 2] - box[0, 0] > (hw[1] - box[0, 2]) * 2: edge = 'r'
	elif box[0, 3] - box[0, 1] > box[0, 1] * 2:           edge = 't'
	elif box[0, 2] - box[0, 0] > box[0, 0] * 2:           edge = 'l'
	elif box[0, 3] - box[0, 1] > (hw[0] - box[0, 3]) * 2: edge = 'b'
	return edge

class Track(object):
	"""This class contains all necessary for every individual track."""
	def __init__(self, pos, score, track_id, features, max_features_num, hw, back=5):
		self.hw = hw
		self.id = track_id
		self.pos = pos
		self.score = score
		self.features = deque([features])
		self.count_inactive = 0
		self.max_features_num = max_features_num
		self.last_pos = deque([pos.clone()], maxlen=back)
		self.last_v = torch.Tensor([])

	def has_positive_area(self):
		# is x2 > x1 + target and y2 > y1 + target
		size_x = self.pos[0, 2] > self.pos[0, 0] + self.hw[1] / (100/1) # percentage
		size_y = self.pos[0, 3] > self.pos[0, 1] + self.hw[0] / (100/1) # percentage
		return size_x and size_y
	
	def isin_edge(self):
		self.edge = isin_edge(self.pos, self.hw)
		return self.edge

	def add_features(self, features):
		"""Adds new appearance features to the object."""
		self.features.append(features)
		if len(self.features) > self.max_features_num:
			self.features.popleft()

	def test_features(self, test_features):
		"""Compares test_features to features of this Track object"""
		if len(self.features) > 1:
			features = torch.cat(list(self.features), dim=0)
		else:
			features = self.features[0]
		# features = features.mean(0, keepdim=True)
		dist = F.pairwise_distance(features, test_features, keepdim=True)
		return dist.mean(0, keepdim=True)
		# return dist

	def reset_last_pos(self):
		self.last_pos.clear()
		self.last_pos.append(self.pos.clone())

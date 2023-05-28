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
	def __init__(self, detector, reid_network, tracker_cfg):
		self.detector = detector
		self.reid_network = reid_network
		self.detection_person_thresh = tracker_cfg.DETECTION_PERSON_THRESH
		self.regression_person_thresh = tracker_cfg.REGRESSION_PERSON_THRESH
		self.detection_nms_thresh = tracker_cfg.DETECTION_NMS_THRESH
		self.regression_nms_thresh = tracker_cfg.REGRESSION_NMS_THRESH

		self.nms_thresh = self.detection_nms_thresh

		self.inactive_patience = tracker_cfg.INACTIVE_PATIENCE
		self.do_reid = tracker_cfg.DO_REID
		self.max_features_num = tracker_cfg.MAX_FEATURES_NUM
		self.reid_sim_threshold = tracker_cfg.REID_SIM_THRESHOLD
		self.reid_iou_threshold = tracker_cfg.REID_IOU_THRESHOLD
		self.do_align = tracker_cfg.DO_ALIGN
		self.motion_model_cfg = tracker_cfg.MOTION_MODEL

		self.warp_mode = eval(tracker_cfg.WARP_MODE)
		self.number_of_iterations = tracker_cfg.NUMBER_OF_ITERATIONS
		self.termination_eps = tracker_cfg.TERMINATION_EPS

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

	def add(self, new_det_pos, new_det_scores, new_det_features):
		"""Initializes new Track objects and saves them."""
		num_new = new_det_pos.size(0)
		for i in range(num_new):
			self.tracks.append(Track(
				new_det_pos[i].view(1, -1),
                new_det_scores[i],
                self.track_num + i,
                new_det_features[i].view(1, -1),
                self.max_features_num,
                self.motion_model_cfg.N_STEPS if self.motion_model_cfg.N_STEPS > 0 else 1
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

	def get_features(self):
		"""Get the features of all active tracks."""
		if len(self.tracks) == 1:
			features = self.tracks[0].features
		elif len(self.tracks) > 1:
			features = torch.cat([t.features for t in self.tracks], 0)
		else:
			features = torch.zeros(0).cuda()
		return features

	def get_inactive_features(self):
		"""Get the features of all inactive tracks."""
		if len(self.inactive_tracks) == 1:
			features = self.inactive_tracks[0].features
		elif len(self.inactive_tracks) > 1:
			features = torch.cat([t.features for t in self.inactive_tracks], 0)
		else:
			features = torch.zeros(0).cuda()
		return features

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
					pos.append(t.pos)
				if len(dist_mat) > 1:
					dist_mat = torch.cat(dist_mat, 0)
					pos = torch.cat(pos, 0)
				else:
					dist_mat = dist_mat[0]
					pos = pos[0]
				# calculate IoU distances
				iou = box_iou(pos, new_det_pos)
				iou_mask = torch.ge(iou, self.reid_iou_threshold)
				iou_neg_mask = ~iou_mask
				# make all impossible assignments to the same add big value
				dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float() * 1000
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

	def get_appearances(self, blob):
		"""Uses the siamese CNN to get the features for all active tracks."""
		pos, _ = self.get_pos()
		new_features = self.reid_network.test_rois(blob['img'], pos).data
		return new_features

	def add_features(self, new_features):
		"""Adds new appearance features to active tracks."""
		for t, f in zip(self.tracks, new_features):
			t.add_features(f.view(1, -1))

	def align(self, blob):
		"""Aligns the positions of active and inactive tracks depending on camera motion."""
		if self.im_index > 0:
			im1 = np.transpose(self.last_image.cpu().numpy(), (1, 2, 0))
			im2 = np.transpose(blob['img'][0].cpu().numpy(), (1, 2, 0))
			im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
			im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
			warp_matrix = np.eye(2, 3, dtype=np.float32)
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.number_of_iterations,  self.termination_eps)
			# https://kite.com/python/docs/cv2.findTransformECC
			# im2 warped to im1 based on similarity in image intensity
			cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, self.warp_mode, criteria)
			warp_matrix = torch.from_numpy(warp_matrix)

			for t in self.tracks:
				t.pos = warp_pos(t.pos, warp_matrix)

			if self.do_reid:
				for t in self.inactive_tracks:
					t.pos = warp_pos(t.pos, warp_matrix)

			if self.motion_model_cfg.ENABLED:
				for t in self.tracks:
					for i in range(len(t.last_pos)):
						t.last_pos[i] = warp_pos(t.last_pos[i], warp_matrix)

	def motion_step(self, track):
		"""Updates the given track's position by one step based on track.last_v"""
		if self.motion_model_cfg.CENTER_ONLY:
			center_new = get_center(track.pos) + track.last_v
			track.pos = make_pos(*center_new, get_width(track.pos), get_height(track.pos))
		else:
			track.pos = track.pos + track.last_v

	def motion(self):
		"""Applies a simple linear motion model that considers the last n_steps steps."""
		for t in self.tracks:
			last_pos = list(t.last_pos)
			# avg velocity between each pair of consecutive positions in t.last_pos
			if self.motion_model_cfg.CENTER_ONLY:
				vs = [get_center(p2) - get_center(p1) for p1, p2 in zip(last_pos, last_pos[1:])]
			else:
				vs = [p2 - p1 for p1, p2 in zip(last_pos, last_pos[1:])]

			t.last_v = torch.stack(vs).mean(dim=0)
			self.motion_step(t)

		if self.do_reid:
			for t in self.inactive_tracks:
				if t.last_v.nelement() > 0:
					self.motion_step(t)

	def step(self, blob, detections):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
		for t in self.tracks:
			# add current position to last_pos list. BH 1
			t.last_pos.append(t.pos.clone())

		###########################
		#  Filter new detections  #  BH 2
		###########################

		boxes, scores, labels = detections['boxes'], detections['scores'], detections['labels']
		boxes = boxes[labels == 1]   # class id
		scores = scores[labels == 1] # class id                                       # by class id
		if boxes.nelement() > 0:
			boxes = clip_boxes_to_image(boxes, blob['img'].shape[-2:])         
			inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)  # by confidence threshold
		else:
			inds = torch.zeros(0).cuda()

		if inds.nelement() > 0:
			det_pos = boxes[inds]
			det_scores = scores[inds]
		else:
			det_pos = torch.zeros(0).cuda()
			det_scores = torch.zeros(0).cuda()

		##################
		# Predict tracks #  BH 3
		##################

		if len(self.tracks):
			# align (BH: camera movement compensation)
			if self.do_align:
				self.align(blob)
			
			# apply motion model if enabled
			if self.motion_model_cfg.ENABLED: 
				# Siddique TODO: tracks should contain position and corresponding center motion: vx, vy
				self.motion()
				self.tracks = [t for t in self.tracks if t.has_positive_area()]

			# regress 
			self.regress_tracks(blob)

		#############################
		# ReID or Create new tracks #  BH 4 (Merged steps 3 and 4)
		#############################

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
					supress = torch.ge(ious[idx], self.nms_thresh)
					for i, sup in enumerate(supress):
						if sup.item() and i != idx.item() and pos_type[i] == 't':
							replaced = True
							trk_pos_keep[pos_idxs[i]] = True
							self.tracks[pos_idxs[i]].pos = poses[idx].view(1, -1)
							self.tracks[pos_idxs[i]].score = scores[idx]
					if not replaced:
						det_pos_keep[pos_idxs[idx]] = True
				else: # existing track
					if scores[idx].item() > self.regression_person_thresh:
						trk_pos_keep[pos_idxs[idx]] = True
			det_pos = det_pos[det_pos_keep]
			det_scores = det_scores[det_pos_keep]
			self.tracks_to_inactive([t for i, t in enumerate(self.tracks) if not trk_pos_keep[i].item()])

		elif det_pos.nelement() > 0:
			# BH: filtering by nms
			# BH: NMS between raw detections to filter
			keep = nms(det_pos, det_scores, self.nms_thresh)
			det_pos = det_pos[keep]
			det_scores = det_scores[keep]


		# ReID and/or create new tracks
		if det_pos.nelement() > 0:
			new_det_pos = det_pos
			new_det_scores = det_scores

			# try to reidentify tracks, BH 5
			# Siddique TODO: tracks should contain position and corresponding center motion: vx, vy and appearance feature descriptor
			new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

			# add new: use global track increment for multi-cameras: same id in different camera is not allowed
			# add new track starting time, BH 6
			if new_det_pos.nelement() > 0:
				self.add(new_det_pos, new_det_scores, new_det_features)

		####################
		# Generate Results #
		# Siddique TODO: Results should contain pos
		# Siddique TODO: Results should contain motion v_cx, v_cy
		# Siddique TODO: Results should contain appearance descriptor
		####################

		for t in self.tracks:
			if t.id not in self.results.keys():
				self.results[t.id] = {}

			# save track state with camera index: without velocity and appearance feature
			self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu(), np.array([t.score.cpu()])])

		for t in self.inactive_tracks:
			t.count_inactive += 1

		self.inactive_tracks = [t for t in self.inactive_tracks if t.has_positive_area() and t.count_inactive <= self.inactive_patience]
		self.im_index += 1
		self.last_image = blob['img'][0]

	def get_results(self):
		return self.results


class Track(object):
	"""This class contains all necessary for every individual track."""
	def __init__(self, pos, score, track_id, features, max_features_num, mm_steps):
		self.id = track_id
		self.pos = pos
		self.score = score
		self.features = deque([features])
		# self.ims = deque([])
		self.count_inactive = 0
		self.max_features_num = max_features_num
		self.last_pos = deque([pos.clone()], maxlen=mm_steps + 1)
		self.last_v = torch.Tensor([])
		self.gt_id = None

	def has_positive_area(self):
		# is x2>x1 and y2>y1
		# Siddique TODO: thresholded by target area (15 x 15)
		return self.pos[0, 2] > self.pos[0, 0] + 15 and self.pos[0, 3] > self.pos[0, 1] + 15

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
		features = features.mean(0, keepdim=True)
		dist = F.pairwise_distance(features, test_features, keepdim=True)
		return dist

	def reset_last_pos(self):
		self.last_pos.clear()
		self.last_pos.append(self.pos.clone())

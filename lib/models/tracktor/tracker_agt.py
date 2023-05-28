import torch
import numpy as np
from .slam import SLAMProcess
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import clip_boxes_to_image, nms
from ...utils.geom_utils import estimate_euclidean_transform
torch.set_printoptions(sci_mode=False)


class Tracker:
	def __init__(self, detector, cfg, tracker_cfg):
		self.detector = detector
		self.detection_person_thresh = tracker_cfg.DETECTION_PERSON_THRESH
		self.regression_person_thresh = tracker_cfg.REGRESSION_PERSON_THRESH
		self.nms_thresh = tracker_cfg.NMS_THRESH
		self.factor = tracker_cfg.REDUCTION_FACTOR
		self.ransac = tracker_cfg.RANSAC_THRES
		self.slam = SLAMProcess(cfg, self.factor, self.ransac)
		self.slam.start()

		self.tracks = []
		self.inactive_tracks = []
		self.cam_oxy = torch.tensor([[0], [0], [1]]).cuda().float()  # initial position of the camera
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

	def tracks_to_inactive(self, tracks):
		self.tracks = [t for t in self.tracks if t not in tracks]
		self.inactive_tracks += tracks

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
				if scores[i] < np.spacing(0):
					self.tracks_to_inactive([t])

	def reid(self, det_pos, det_scores):
		# For active tracks
		if len(self.tracks):
			cxy_t, cxy_f = [], []
			for t in self.tracks:
				if not t.isin_edge() and t.cxy != None:      # not in edge and not for the first time
					cxy = (t.pos[:, :2] + t.pos[:, 2:]) / 2  # center in the current frame
					cxy_f.append(cxy[0].tolist())
					cxy_t.append(t.cxy[0].tolist())
			if len(cxy_f) > 1:                                   # Trust current positions and orientations
				Ha = np.eye(3)
				Ha[:2, :2], Ha[:2, 2] = estimate_euclidean_transform(np.array(cxy_f), np.array(cxy_t))
				self.slam.H = Ha

			self.cam_oxy = torch.tensor(self.slam.H).cuda().float()[:, -1:]

			for t in self.tracks:
				if not t.isin_edge() and t.cxy == None: # not in edge first time (no cen and rad yet)
					t.cxy = (t.pos[0, :2] + t.pos[0, 2:]) / 2 + self.cam_oxy[0:2, 0].view(1, 2) # center
					t.rad = sum(t.pos[0, 2:] - t.pos[0, :2]) / 4                                # radious

		else: self.cam_oxy = torch.tensor(self.slam.H).cuda().float()[:, -1:]

		# For inactive tracks
		cxy = []
		for t in self.inactive_tracks:
			if t.cxy != None: cxy.append(t.cxy)
			else: cxy.append((t.pos[0, :2] + t.pos[0, 2:]) / 2 + t.cam_oxy[0:2, 0].view(1, 2))
		if len(cxy):
			cxy = torch.cat(cxy)
			if det_pos.nelement() > 0:
				det_cxy = (det_pos[:, :2] + det_pos[:, 2:]) / 2  + self.cam_oxy[0:2, 0].view(1, 2)
				dist = torch.cdist(cxy[None, :, :], det_cxy[None, :, :])[0]
				track_ind, det_ind = linear_sum_assignment(dist.cpu().numpy())
				not_assigned = []
				remove_inactive = []
				for tr, dt in zip(track_ind, det_ind):
					print(tr, dt, dist[tr, dt])
					t = self.inactive_tracks[tr]
					if t.rad != None:
						if dist[tr, dt] <= 500: # 2 * t.rad:
							self.tracks.append(t)		
							t.pos = det_pos[dt].view(1, -1)
							remove_inactive.append(t)
						else: not_assigned.append(dt)
					else:
						if dist[tr, dt] <= 500: # sum(det_pos[dt, 2:] - det_pos[dt, :2]) / 2:
							self.tracks.append(t)		
							t.pos = det_pos[dt].view(1, -1)
							remove_inactive.append(t)
						else: not_assigned.append(dt)
				for t in remove_inactive:
					self.inactive_tracks.remove(t)
				det_pos, det_scores = det_pos[not_assigned], det_scores[not_assigned]

		return det_pos, det_scores	


	def add(self, new_det_pos, new_det_scores, hw):
		"""Initializes new Track objects and saves them."""
		num_new = new_det_pos.size(0)
		for i in range(num_new):
			track_num = self.track_num + 1
			pos = new_det_pos[i].view(1, -1)
			score = new_det_scores[i]
			track = Track(pos, self.cam_oxy, score, track_num, hw)
			if track.has_positive_area() and track.isin_edge() or self.im_index == 0:
				self.tracks.append(track)
				self.track_num = track_num

	def step(self, blob, detections):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
		# get image dimensions
		h, w = blob['img'].shape[2:4]
		self.slam.args = (blob['img'], h, w, self.im_index)
		self.slam.run()

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
							# lens.append(len(self.tracks[indexes[-1]].last_pos))   # and lengths (global_pos)
							lens.append(scores[idx].item())                       # temporarily changed to highest score
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

		self.slam.join()
		# Do ReID if activated
		if True: # TODO: add param do-reid
			new_det_pos, new_det_scores = self.reid(det_pos, det_scores)

		# Create new tracks
		if len(new_det_pos):
			if new_det_pos.nelement() > 0:
				self.add(new_det_pos, new_det_scores, (h, w))

		# Generate Results 
		for t in self.tracks:
			if t.id not in self.results.keys():
				self.results[t.id] = {}
			self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu(), np.array([t.score.cpu()])])
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
	def __init__(self, pos, cam_oxy, score, track_id, hw):
		self.hw = hw
		self.pos = pos               # bounding box in the first frame coordinates (and subsequent frame coordinates)
		self.score = score           # score
		self.id = track_id           # ID
		self.cam_oxy = cam_oxy
		if not self.isin_edge():
			self.cxy = (self.pos[0, :2] + self.pos[0, 2:]) / 2 + cam_oxy[0:2, 0].view(1, 2)  # center
			self.rad = sum(self.pos[0, 2:] - self.pos[0, :2]) / 4                          # radious
		else:
			self.cxy = None
			self.rad = None

	def has_positive_area(self):
		# is x2 > x1 + target and y2 > y1 + target
		size_x = self.pos[0, 2] > self.pos[0, 0] + self.hw[1] / (100/2) # percentage
		size_y = self.pos[0, 3] > self.pos[0, 1] + self.hw[0] / (100/2) # percentage
		return size_x and size_y
	
	def isin_edge(self):
		self.edge = isin_edge(self.pos, self.hw)
		return self.edge

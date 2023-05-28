import numpy as np

def estimate_euclidean_transform(keypoints1, keypoints2):
    """
    Estimate Euclidean transformation given 2D keypoint correspondences using least squares method.
    Args:
        keypoints1 (ndarray): Array of shape (N, 2) representing 2D keypoints in the source image.
        keypoints2 (ndarray): Array of shape (N, 2) representing 2D keypoints in the target image.
    Returns:
        R (ndarray): Array of shape (2, 2) representing the estimated rotation matrix.
        t (ndarray): Array of shape (2,) representing the estimated translation vector.
    """
    # Check if the number of keypoints is the same in both images
    assert keypoints1.shape[0] == keypoints2.shape[0], "Number of keypoints must be the same in both images"

    # Find the centroids of the keypoints
    centroid1 = np.mean(keypoints1, axis=0)
    centroid2 = np.mean(keypoints2, axis=0)

    # Center the keypoints around the centroids
    centered_keypoints1 = keypoints1 - centroid1
    centered_keypoints2 = keypoints2 - centroid2

    # Compute the covariance matrix
    covariance_matrix = centered_keypoints1.T @ centered_keypoints2

    # Perform singular value decomposition (SVD)
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Compute the rotation matrix and translation vector
    R = Vt.T @ U.T
    t = centroid2 - R @ centroid1

    return R, t
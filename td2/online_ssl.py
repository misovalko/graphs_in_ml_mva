import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import sys
from scipy.spatial import distance
import scipy.io as sio

path = os.path.dirname(os.getcwd())
sys.path.append(path)
from helper import *
from helper_online_ssl import create_user_profile, load_profile, preprocess_face

face_haar_cascade = cv.CascadeClassifier("data/haarcascade_frontalface_default.xml")
eye_haar_cascade = cv.CascadeClassifier("data/haarcascade_eye.xml")


class IncrementalKCenters:
    def __init__(self, labeled_faces, labels, max_num_centroids=50):
        #  Number of labels to cluster
        self.n_labels = max(labels)

        #  Dimension of the input image
        self.image_dimension = labeled_faces.shape[1]

        #  Check input validity
        assert (set(labels) == set(
            range(1, 1 + self.n_labels))), "Initially provided faces should be labeled in [1, max]"
        assert (len(labeled_faces) == len(labels)), "Initial faces and initial labels are not of same size"

        #  Number of labelled faces
        self.n_labeled_faces = len(labeled_faces)

        # Model parameter : number of maximum stored centroids
        self.max_num_centroids = max_num_centroids

        # Model centroids (inital labeled faces). Shape = (number_of_centroids, dimension)
        self.centroids = labeled_faces

        # Centroids labels
        self.Y = labels

        # Variables that are initialized in online_ssl_update_centroids()
        self.centroids_distances = None
        self.taboo = None
        self.V = None
        self.init = True
        self.last_face = None  # index of x_t (initialized later)
        #
        print('[s]ave a frame ?')

    def online_ssl_update_centroids(self, face):
        """
        TO BE COMPLETED (question 3.1)

        Implements Algorithm 1.
        :param face: the new sample
        """

        assert (self.image_dimension == len(face)), "new image not of good size"

        # Case 1: maximum number of centroids has been reached.
        if self.centroids.shape[0] >= self.max_num_centroids + 1:
            """
            Initialization.
            """
            if self.init:
                #  Compute the centroids distances
                self.centroids_distances = distance.cdist(self.centroids, self.centroids)

                #  set labeled nodes and self loops as infinitely distant
                np.fill_diagonal(self.centroids_distances, +np.Inf)
                self.centroids_distances[0:self.n_labeled_faces, 0:self.n_labeled_faces] = +np.Inf

                # put labeled nodes in the taboo list
                self.taboo = np.array(range(self.centroids.shape[0])) < self.n_labeled_faces

                # initialize multiplicity
                self.V = np.ones(self.centroids.shape[0])
                self.init = False

            """
            Find c_rep and c_add following Algorithm 1.
            
            - c_1, c_2 = two closest centroids (minimum distance) such that at least one of them is not in self.taboo.
            - c_rep = centroid in {c_1, c_2} that is in self.taboo. If none of them is in self.taboo, c_rep is the one
                      with largest multiplicity.
            - c_add = centroid in {c_1, c_2} that is not c_rep.
            """
            min_dist = None
            c_rep, c_add = None, None

            """
            Update data structures: self.centroids and self.V
            """

            # ...

            """
            Update the matrix containing the distances.
            """
            dist_row = distance.cdist(np.array([self.centroids[c_add]]), self.centroids)[0]
            dist_row[c_add] = +np.inf
            self.centroids_distances[c_add, :] = dist_row
            self.centroids_distances[:, c_add] = dist_row
            self.last_face = c_add

        # Case 2: create new centroid with face
        # Remark: the multiplicities vector self.V is initialized in case 1.
        else:
            current_len = len(self.centroids)
            self.Y = np.append(self.Y, 0)
            self.centroids = np.vstack([self.centroids, face])

    def online_ssl_compute_solution(self):
        """
        TO BE COMPLETED. (question 3.2)

        Implements Algorithm 2.

        Returns a prediction corresponding to self.last_face.
        """

        """
        Choose the experiment parameters
        """
        var = 300
        eps = None
        k = None
        laplacian_regularization = None
        laplacian_normalization = ""

        """
        Build graph and its Laplacian
        """
        W = build_similarity_graph(self.centroids, var=var, eps=eps, k=k)
        if self.init:
            V = np.diag(np.ones(self.centroids.shape[0]))
            self.last_face = self.centroids.shape[0] - 1
        else:
            V = np.diag(self.V)
        W = V.dot(W.dot(V))

        # Laplacian
        L = build_laplacian(W, laplacian_normalization)
        # regularized Laplacian
        Q = L + laplacian_regularization*np.eye(W.shape[0])

        """
        Compute the hardHFS solution f. 
        You can adapt the code you wrote in harmonic_function_solution.py.
        """
        f = None

        return f[self.last_face]


def online_face_recognition(profile_names, n_pictures=15):
    """
    TO BE COMPLETED. (question 3.4)

    Run online face recognition.
    :param profile_names: user names used in create_user_profile()
    :param n_pictures: number of pictures to use for each user_name
    """
    images = []
    labels = []
    label_names = []
    for i, name in enumerate(profile_names):
        p = load_profile(name)
        p = p[0:n_pictures, ]
        images += [p]
        labels += [np.ones(p.shape[0]) * (i + 1)]
        label_names += [name]
    faces = np.vstack(images)
    labels = np.hstack(labels).astype(np.int)
    #  Generate model
    model = IncrementalKCenters(faces, labels)
    # Start camera
    cam = cv.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        grey_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        working_image = cv.bilateralFilter(grey_image, 9, 75, 75)
        working_image = cv.equalizeHist(working_image)
        working_image = cv.GaussianBlur(working_image, (5, 5), 0)
        box = face_haar_cascade.detectMultiScale(working_image)
        for b0 in box:
            x, y = b0[0], b0[1]
            x_range, y_range = b0[2], b0[3]
            # look for eye classifier
            local_image = img[y:(y + y_range), x:(x + x_range)]
            eye_box = eye_haar_cascade.detectMultiScale(local_image)
            if len(eye_box) == 0:
                cv.rectangle(img, tuple([b0[0] - 4, b0[1] - 4]), tuple([b0[0] + b0[2] + 4, b0[1] + b0[3] + 4]),
                             (0, 0, 255), 2)
                continue
            # select face
            local_image = grey_image[y:(y + y_range), x:(x + x_range)]
            x_t = preprocess_face(local_image)

            """
            Centroids are updated here
            """
            model.online_ssl_update_centroids(x_t)
            p1, p2 = tuple([b0[0] - 4, b0[1] - 4]), tuple([b0[0] + b0[2] + 4, b0[1] + b0[3] + 4])

            """
            HardHFS solution is computed here
            """
            f = model.online_ssl_compute_solution()
            lab = np.argsort(f)

            """
            Change False by something else to be able to disregard faces it cannot recognize (question 3.4)
            """
            if False:
                color = (100, 100, 100)
                txt = "unknown"
                cv.putText(img, txt, (p1[0], p1[1] - 5), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, color)
            else:
                for i, l in enumerate(lab):
                    color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)][l]
                    txt = label_names[l] + "  " + ('%.4f' % np.abs(f[l]))
                    cv.putText(img, txt, (p1[0], p1[1] - 5 - 10 * i), cv.FONT_HERSHEY_COMPLEX_SMALL,
                               0.5 + 0.5 * (i == f.shape[0] - 1), color)
            cv.rectangle(img, p1, p2, color, 2)
        cv.imshow("cam", img)
        key = cv.waitKey(1)
        if key in [27, 101]:
            break
        if key == ord('s'):
            # Save face
            print('saved')
            cv.imwrite("frame.png", img)
            ## cv.waitKey(1)
    cv.destroyAllWindows()


if __name__ == '__main__':
    create_user_profile('test_username')


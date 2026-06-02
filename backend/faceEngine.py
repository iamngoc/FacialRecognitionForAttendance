"""
model view the photo
compute 512-dimensional feature vector for the face
compare with all stored vectors in database
if vectors show similarity > threshold -> same person
"""

# import datetime
# import os
import logging
import numpy as np
from numpy.linalg import norm
from typing import Optional
import cv2
# import insightface
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

class FacialRecognizingEngine:
    """
    Manage the ArcFace model and all facial operations
    """
    def __init__(self,
                 threshold: float = 0.45, # 0.45=45% similarity
                 model_name: str = "buffalo_l"):
        self.threshold = threshold # high thres=stricter, low thres=toleranter
        self.model = None
        self.model_name = model_name

    def _load_model(self, model_name):
        """
        load the pretrained model Arcface of InsightFace
        """
        try:
            logger.info(f"Loading ArcFace model '{model_name}'...")
            self.model = FaceAnalysis(name=model_name, provider=['GPUExecutionProvider'])
            self.model.prepare(ctx_id=0, det_size=(640, 640))

            logger.info(f"ArcFace model '{model_name}' successfully loaded.")
            logger.info(f"threshold: '{self.threshold}'")

        except Exception as e:
            logger.error(f"ArcFace model '{model_name}' failed to load: {e}")
            raise

    def compute_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        take a photo and compute the 512-dimensional embedding ("facial fingerprint")
        :param image:
        :return: numpy array with 512 number or None when no photo detected
        """
        if image is None or image.size == 0:
            return None

        try:
            face_detection = self.model.get(image)

            if not face_detection:
                logger.error(f"No face detected in image '{image}'")
                return None

            # when more than one face detected, take the biggest face
            if len(face_detection) > 1:
                biggest_face = max(face_detection, key=lambda g: (g.bbox[2]-g.bbox[0]) * (g.bbox[3]-g.bbox[1]))
                return biggest_face.embedding
            else:
                return face_detection[0].embedding

        except Exception as e:
            logger.error(f"ArcFace model '{image}' failed to compute embedding: {e}")
            return None

    def cos_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        compute how similar two embeddings are
        each embedding is a 512-dimensional vector, when 2 vectors show the same direction -> one person
        :param embedding1:
        :param embedding2:
        :return: number between 0 and 1 with 0=different and 1=same
        """
        n1 = norm(embedding1)
        n2 = norm(embedding2)
        if n1 == 0 or n2 == 0:
            return 0

        return float(np.dot(embedding1, embedding2)) / (n1 * n2)

    def recognize_person(self,
                         camera_photo: np.ndarray,
                         employee_embeddings: list[dict])-> Optional[dict]:
        """
        recognize person
        :param camera_photo: photo of person
        :param employee_embeddings: list of {'id': int, 'name': str, 'embedding': np.array[512]
        :return: 'employee id': int, 'name': str, 'score': float or None when no person detected
        """
        # 1. compute embedding of current photo
        camera_embedding = self.compute_embedding(camera_photo)
        if camera_embedding is None:
            return None # no face detected
        # 2. compare with all employees in database
        best_target = None
        best_score = 0.0

        for employee in employee_embeddings:
            if employee['embedding'] is None:
                continue # no embedding stored -> skip

            score = self.cos_similarity(camera_embedding, np.array(employee['embedding']))

            logger.debug(f"Comparing with employee {employee['name']}: "
                         f"score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_target = employee

        # 3. is the AI secure enough?
        if best_score >= self.threshold:
            logger.info(f"Person {best_target['name']} detected"
                        f" with score: {best_score:.4f}")

            return {'mitarbeiter_id': best_target['id'],
                    'name': best_target['name'],
                    'score': best_score,
                    'embedding': camera_embedding}
        else:
            logger.info(f"WARNING: unrecognized Person detected"
                        f" {best_score:.4f} < {self.threshold:.4f}" )
            return None

    def mark_face_in_photo(self,
                           photo: np.ndarray,
                           name: str = 'unknown',
                           score: float = 0.0) -> np.ndarray:
        """
        draw a frame around the known face
        :param photo:
        :param name:
        :param score:
        :return: array
        """
        faces = self.model.get(photo)
        photo_copy = photo.copy()

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # green frame for known person, red for unknown
            color = (0, 255, 0) if name != 'unknown' else (0, 0, 255)

            cv2.rectangle(photo_copy, (x1, y1), (x2, y2), color, 2)

            # show name and score
            show_text = f'name: {name} (score: {score:.4f})' if score > 0 else None
            cv2.putText(photo_copy, show_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return photo_copy

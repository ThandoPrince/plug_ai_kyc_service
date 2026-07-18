"""
Computer-vision checks for affidavit documents: image quality (blur,
brightness), orientation, and presence of expected physical markings
(signature, commissioner's stamp).

These are heuristic checks, not an authenticity guarantee — they catch
obviously unusable scans and flag documents worth a closer look. They
are not a substitute for human review of borderline cases.
"""

from typing import Union

import cv2
import numpy as np

from src.api.utils.image_utils import resize_max_dimension


class AffidavitVision:
    def __init__(self, image: Union[str, np.ndarray]):
        """
        Args:
            image: either a path to an image file, or an already-loaded
                   OpenCV BGR array (e.g. a page from DocumentProcessor).
        """
        if isinstance(image, str):
            loaded = cv2.imread(image)
            if loaded is None:
                raise ValueError(f"Unable to load affidavit image: {image}")
            self.image = loaded
        else:
            if image is None or image.size == 0:
                raise ValueError("Unable to load affidavit: empty image array.")
            self.image = image

        self.image = resize_max_dimension(self.image, max_dimension=2000)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def detect_blur(self) -> dict:
        variance = float(cv2.Laplacian(self.gray, cv2.CV_64F).var())
        return {
            "passed": variance > 120,
            "score": variance,
        }

    def detect_brightness(self) -> dict:
        brightness = float(np.mean(self.gray))
        return {
            "passed": 60 < brightness < 220,
            "score": brightness,
        }

    def detect_rotation(self) -> dict:
        """
        Estimate document skew from dominant line angles, rather than just
        checking whether any lines exist — "lines are present" doesn't
        tell you if the page is actually straight.
        """
        edges = cv2.Canny(self.gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None:
            return {
                "passed": False,
                "skew_degrees": None,
                "detail": "No dominant lines detected.",
            }

        angles = []
        for line in lines:
            rho, theta = line[0]
            angle_degrees = (theta * 180 / np.pi) - 90
            # Fold into [-45, 45] so both near-horizontal and near-vertical
            # dominant lines are treated as orientation evidence.
            if angle_degrees > 45:
                angle_degrees -= 90
            elif angle_degrees < -45:
                angle_degrees += 90
            angles.append(angle_degrees)

        skew = float(np.median(angles))

        return {
            "passed": abs(skew) <= 5,
            "skew_degrees": skew,
        }

    def detect_signature_regions(self) -> list:
        thresh = cv2.threshold(self.gray, 180, 255, cv2.THRESH_BINARY_INV)[1]

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        signatures = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Loose heuristic: signatures tend to be irregular, elongated
            # blobs in this size/aspect range. Not a real signature
            # classifier — flags candidate regions, doesn't verify a
            # signature is genuinely present or whose it is.
            if 500 < area < 50000:
                aspect_ratio = w / h if h else 0
                if 1.2 < aspect_ratio < 8:
                    signatures.append({"x": x, "y": y, "width": w, "height": h})

        return signatures

    def detect_stamp(self) -> dict:
        circles = cv2.HoughCircles(
            self.gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=150,
            param1=100,
            param2=30,
            minRadius=30,
            maxRadius=250,
        )

        return {
            "detected": circles is not None,
            "count": 0 if circles is None else int(circles.shape[1]),
        }

    def detect_document_edges(self) -> dict:
        edges = cv2.Canny(self.gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {"passed": False, "area": 0.0, "detail": "No contours detected."}

        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))

        return {
            "passed": area > 100000,
            "area": area,
        }

    def analyse(self) -> dict:
        return {
            "blur": self.detect_blur(),
            "brightness": self.detect_brightness(),
            "rotation": self.detect_rotation(),
            "stamp": self.detect_stamp(),
            "signatures": self.detect_signature_regions(),
            "document_edges": self.detect_document_edges(),
        }
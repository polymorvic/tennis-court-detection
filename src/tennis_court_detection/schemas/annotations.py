from pydantic import BaseModel, Field
from enum import StrEnum
from cvgeomkit.geometry import Point

import numpy as np


class TennisCourtKeyPointLabel(StrEnum):
    left_outer_baseline_point = "left_outer_baseline_point"
    left_outer_baseline_point_opposite = "left_outer_baseline_point_opposite"
    left_inner_baseline_point = "left_inner_baseline_point"
    left_inner_baseline_point_opposite = "left_inner_baseline_point_opposite"
    left_outer_netline_point = "left_outer_netline_point"
    left_inner_netline_point = "left_inner_netline_point"
    right_outer_baseline_point = "right_outer_baseline_point"
    right_outer_baseline_point_opposite = "right_outer_baseline_point_opposite"
    right_inner_baseline_point = "right_inner_baseline_point"
    right_inner_baseline_point_opposite = "right_inner_baseline_point_opposite"
    right_outer_netline_point = "right_outer_netline_point"
    right_inner_netline_point = "right_inner_netline_point"
    left_service_point = "left_service_point"
    left_service_point_opposite = "left_service_point_opposite"
    right_service_point = "right_service_point"
    right_service_point_opposite = "right_service_point_opposite"
    left_service_netline_point = "left_service_netline_point"
    right_service_netline_point = "right_service_netline_point"
    left_centre_service_point = "left_centre_service_point"
    left_centre_service_point_opposite = "left_centre_service_point_opposite"
    right_centre_service_point = "right_centre_service_point"
    right_centre_service_point_opposite = "right_centre_service_point_opposite"
    left_top_netline_point = "left_top_netline_point"
    right_top_netline_point = "right_top_netline_point"
    middle_top_netline_point = "middle_top_netline_point"


    @classmethod
    def names(cls):
        return [item.name for item in cls]


class ImageMetaData(BaseModel):
    name: str
    width: int
    height: int
    file_origin: str


class KeyPointCoordinates(BaseModel):
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y])


class KeyPointAnnotation(BaseModel):
    label: TennisCourtKeyPointLabel
    coordinates: KeyPointCoordinates


class ImageAnnotation(BaseModel):
    image: ImageMetaData
    key_points: list[KeyPointAnnotation]

    def prepare_for_execution(self) -> dict[str, Point]:
        x_w = self.image.width
        x_h = self.image.height

        gt = {}
        for kp in self.key_points:
            name = kp.label.name
            kp_x = kp.coordinates.x
            kp_y = kp.coordinates.y
            gt[name] = Point(int(kp_x * x_w / 100), int(kp_y * x_h / 100))

        return gt

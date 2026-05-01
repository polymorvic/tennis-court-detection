from pydantic import BaseModel, Field
from enum import StrEnum


class TennisCourtKeyPointLabel(StrEnum):
    left_outer_netline_point = "left_outer_netline_point"
    left_inner_netline_point = "left_inner_netline_point"
    right_inner_netline_point = "right_inner_netline_point"
    right_outer_netline_point = "right_outer_netline_point"
    centre_service_point = "centre_service_point"
    netline_service_point = "netline_service_point"
    left_service_point = "left_service_point"
    right_service_point = "right_service_point"
    left_outer_baseline_point = "left_outer_baseline_point"
    left_inner_baseline_point = "left_inner_baseline_point"
    right_inner_baseline_point = "right_inner_baseline_point"
    right_outer_baseline_point = "right_outer_baseline_point"


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


class KeyPointAnnotation(BaseModel):
    label: TennisCourtKeyPointLabel
    coordinates: KeyPointCoordinates


class ImageAnnotation(BaseModel):
    image: ImageMetaData
    key_points: list[KeyPointAnnotation]

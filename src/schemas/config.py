from enum import StrEnum

from pydantic import BaseModel, model_validator


class ServiceSide(StrEnum):
    LEFT = 'left'
    RIGHT = 'right'


class MatchType(StrEnum):
    SINGLES = "singles"
    DOUBLES = "doubles"


class MatchParams(BaseModel):
    service_side: ServiceSide
    match_type: MatchType


class TraversingParams(BaseModel):
    roi_height_px: int
    step_px: int
    warmup: int


class LineDetectionParams(BaseModel):
    canny_lower_thresh: int
    canny_upper_thresh: int
    hough_thresh: int
    min_line_len_ratio: float
    min_line_gap_px: int
    vertical_center_delta_px: int
    white_line_bin_lower_thresh: int
    white_line_bin_upper_thresh: int

    @model_validator(mode="after")
    def validate_canny_thresholds(self):
        if self.canny_lower_thresh >= self.canny_upper_thresh:
            raise ValueError(
                "canny_lower_thresh must be smaller than "
                "canny_upper_thresh"
            )

        return self


class DetectionParams(BaseModel):
    traversing: TraversingParams
    lines_detection: LineDetectionParams


class Params(BaseModel):
    match_params: MatchParams
    detection_params: DetectionParams

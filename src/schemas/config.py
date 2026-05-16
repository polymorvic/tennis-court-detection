from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


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
    roi_height_px: int = Field(gt = 0)
    step_px: int = Field(gt = 0)
    warmup: int = Field(ge = 0)


class LineDetectionParams(BaseModel):
    canny_lower_thresh: int = Field(ge = 0)
    canny_upper_thresh: int = Field(ge = 0)
    hough_thresh: int = Field(ge = 0)
    min_line_len_ratio: float = Field(ge = 0)
    min_line_gap_px: int = Field(ge = 0)
    vertical_center_delta_px: int = Field(ge = 0)
    white_line_bin_lower_thresh: int = Field(ge = 0, le=255)
    white_line_bin_upper_thresh: int = Field(ge = 0, le=255)


    @model_validator(mode="after")
    def validate_thresholds(self):
        if self.canny_lower_thresh >= self.canny_upper_thresh:
            raise ValueError(
                "canny_lower_thresh must be smaller than "
                "canny_upper_thresh"
            )

        if self.white_line_bin_lower_thresh > self.white_line_bin_upper_thresh:
            raise ValueError(
                "white_line_bin_lower_thresh must be smaller "
                "than or equal to white_line_bin_upper_thresh"
            )
        
        return self


class DetectionParams(BaseModel):
    traversing: TraversingParams
    lines_detection: LineDetectionParams


class Params(BaseModel):
    match_params: MatchParams
    detection_params: DetectionParams


class PicsBlacklist(BaseModel):
    blacklist: list[str]

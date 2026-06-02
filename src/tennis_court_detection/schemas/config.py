from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class ServiceSide(StrEnum):
    LEFT = 'left'
    RIGHT = 'right'


class MatchType(StrEnum):
    SINGLES = "singles"
    DOUBLES = "doubles"


class Surface(StrEnum):
    CLAY = "clay"
    OTHER = "other"


class MatchParams(BaseModel):
    service_side: ServiceSide
    match_type: MatchType
    surface: Surface


class BasicParams(BaseModel):
    roi_height_ratio: float = Field(default=0.075, gt=0, le=1.0)
    step_height_ratio: float = Field(default=0.02, gt=0, le=0.1)
    crop_center_width_ratio: float = Field(default=0.4, gt=0, le=1.0)


class BaselineParams(BaseModel):
    warmup_height_ratio: float = Field(default=0.1, ge=0, le=0.5)
    canny_lower_thresh: int = Field(default=20, ge=0)
    canny_upper_thresh: int = Field(default=100, ge=0)
    canny_lower_thresh_offset: int = Field(default=80, ge=0)
    canny_upper_thresh_offset: int = Field(default=100, ge=0)
    hough_thresh: int = Field(default=50, gt=0)
    hough_thresh_offset: int = Field(default=-10, le=0)

    min_line_len_width_ratio: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
    )
    min_line_len_ensure_width_ratio: float = Field(
        default=0.03,
        gt=0.0,
        le=0.05
    )
    max_line_gap_width_ratio: float = Field(default=0.005, ge=0)
    horizontal_line_slope_tolerance: float = Field(default=0.03, ge=0)
    delta_ensure_height_ratio: float = Field(default=0.1, ge=0)

    @model_validator(mode="after")
    def validate_thresholds(self):
        if self.canny_lower_thresh >= self.canny_upper_thresh:
            raise ValueError(
                "canny_lower_thresh must be smaller than "
                "canny_upper_thresh"
            )
        return self


class DetectionParams(BaseModel):
    basic: BasicParams
    baseline: BaselineParams


class Params(BaseModel):
    match_params: MatchParams
    detection_params: DetectionParams


class PicsBlacklist(BaseModel):
    blacklist: list[str]

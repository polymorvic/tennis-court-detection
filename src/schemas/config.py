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


class BasicParams(BaseModel):
    roi_h_px: int = Field(default=80, gt=0)
    step_px: int = Field(default=20, gt=0)
    crop_center_ratio: float = Field(default=0.4, gt=0, le=1.0)


class BaselineParams(BaseModel):
    warmup: int = Field(default=15, ge=0)
    canny_lower_thresh: int = Field(default=20, ge=0)
    canny_upper_thresh: int = Field(default=100, ge=0)
    hough_thresh: int = Field(default=100, gt=0)
    min_line_len_ratio: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
    )
    min_line_len_ensure_ratio: float = Field(
        default=0.01,
        gt=0.0,
        le=0.05
    )
    min_line_gap_px: int = Field(default=10, gt=0)
    h_line_slope_tolerance: float = Field(default=0.03, ge=0)

    @model_validator(mode="after")
    def validate_thresholds(self):
        if self.canny_lower_thresh >= self.canny_upper_thresh:
            raise ValueError(
                "canny_lower_thresh must be smaller than "
                "canny_upper_thresh"
            )
        return self


# class LineDetectionParams(BaseModel):
#     canny_lower_thresh: int = Field(ge = 0)
#     canny_upper_thresh: int = Field(ge = 0)
#     hough_thresh: int = Field(ge = 0)
#     min_line_len_ratio: float = Field(ge = 0)
#     min_line_gap_px: int = Field(ge = 0)
#     vertical_center_delta_px: int = Field(ge = 0)
#     white_line_bin_lower_thresh: int = Field(ge = 0, le=255)
#     white_line_bin_upper_thresh: int = Field(ge = 0, le=255)
#     white_line_bin_upper_thresh: int = Field(ge = 0, le=255)
#     max_spread_vlines_px: int

#     @model_validator(mode="after")
#     def validate_thresholds(self):
#         if self.canny_lower_thresh >= self.canny_upper_thresh:
#             raise ValueError(
#                 "canny_lower_thresh must be smaller than "
#                 "canny_upper_thresh"
#             )

#         if self.white_line_bin_lower_thresh > self.white_line_bin_upper_thresh:
#             raise ValueError(
#                 "white_line_bin_lower_thresh must be smaller "
#                 "than or equal to white_line_bin_upper_thresh"
#             )
        
#         return self


class DetectionParams(BaseModel):
    basic: BasicParams
    baseline: BaselineParams


class Params(BaseModel):
    match_params: MatchParams
    detection_params: DetectionParams


class PicsBlacklist(BaseModel):
    blacklist: list[str]

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceTennisCourtDimensions:
	width: int = 10_973
	length: int = 23_770
	dist_outer_sideline: int = 1_372
	dist_from_baseline: int = 5_485

	@property
	def court_length_half(self) -> int:
		return self.length // 2

	@property
	def court_width_half(self) -> int:
		return self.width // 2


COURT_DIMENSIONS = ReferenceTennisCourtDimensions()


CLOSER_SIDE_POINTS = [
	'left_service_point',
	'right_service_point',
	'left_outer_baseline_point',
	'left_inner_baseline_point',
	'right_inner_baseline_point',
	'right_outer_baseline_point',
	'left_centre_service_point',
	'right_centre_service_point',
]


NET_POINTS = [
	'left_outer_netline_point',
	'left_inner_netline_point',
	'right_inner_netline_point',
	'right_outer_netline_point',
	'left_service_netline_point',
	'right_service_netline_point',
	'left_top_netline_point',
	'middle_top_netline_point',
	'right_top_netline_point'
]


OPPOSITE_SIDE_POINTS = [
	'left_service_point_opposite',
	'left_centre_service_point_opposite',
	'right_centre_service_point_opposite',
	'right_service_point_opposite',
	'right_outer_baseline_point_opposite',
	'right_inner_baseline_point_opposite',
	'left_inner_baseline_point_opposite',
	'left_outer_baseline_point_opposite'
]
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


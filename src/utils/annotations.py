from pathlib import Path
import json
from typing import ClassVar, Literal, Self, Type

import matplotlib.pyplot as plt
import numpy as np
from src.schemas.annotations import ImageAnnotation, TennisCourtKeyPointLabel
from cvgeomkit.geometry.points import Point
from cvgeomkit.common import ArrayLike
    

def transform_keypoint_annotation(
    img: ArrayLike, 
    annotation: list[list[float]] | np.ndarray
) -> list[Point] | np.ndarray:
    arr = np.array(annotation) * np.array([img.shape[1], img.shape[0]]) / 100
    return Point.from_iterable(*arr) if isinstance(annotation, list) else arr.astype(np.float32)


_RawAnnotations = list[dict[Literal["filename", "data"], str | list]]


class TennisCourtAnnotationCollection[AT: ImageAnnotation]:
    annotation_model: ClassVar[Type[ImageAnnotation]] = ImageAnnotation

    def __init__(self, 
                 root_dir: Path | str, 
                 raw_annotations: _RawAnnotations | None = None,
                 cleaned_annotations: dict[str, AT] | None = None
                ) -> None:
        self._root_dir: Path = Path(root_dir)
        self._raw_annotations = raw_annotations
        self.cleaned_annotations = cleaned_annotations


    @classmethod
    def from_raw_dir(cls, root_dir: Path | str, extension: str = 'json') -> Self:
        obj = cls(root_dir)
        obj._raw_annotations = obj._concat_files(extension)
        obj.cleaned_annotations = obj._clean_annotations()
        return obj


    @classmethod
    def from_clean_file(cls, file_path: Path | str) -> Self:
        file_path = Path(file_path)

        with file_path.open(encoding="utf-8") as f:
            raw_cleaned = json.load(f)

        cleaned = {item['image']['name']: cls.annotation_model.model_validate(item) for item in raw_cleaned}
        return cls(file_path.parent, [], cleaned)



    def _clean_annotations(self) -> dict[str, AT]:
        cleaned = {}

        for bundle in self._raw_annotations:
            file_origin = str(bundle["filename"])

            for task in bundle["data"]:
                anns = task.get("annotations")

                if not anns:
                    continue

                result = anns[0].get("result") or []
                key_points = []

                w = h = None
                for r in result:
                    if r.get("type") != "keypointlabels":
                        continue

                    val = r.get("value") or {}
                    lbl = val.get("keypointlabels")

                    if not lbl:
                        continue

                    if not key_points:
                        w, h = r["original_width"], r["original_height"]

                    key_points.append(
                        {"label": lbl[0], "coordinates": {"x": val["x"], "y": val["y"]}}
                    )

                if not key_points:
                    continue

                name = self._build_image_name(task)
                cleaned[name] = self.annotation_model.model_validate(
                    {
                        "image": {
                            "name": name,
                            "width": w,
                            "height": h,
                            "file_origin": file_origin,
                        },
                        "key_points": key_points,
                    }
                )
                
        return cleaned
    

    def validate(self) -> None:
        if not self.cleaned_annotations:
            raise ValueError("Clean annotations not set")

        for name, ann in self.cleaned_annotations.items():
            labels = [kp.label for kp in ann.key_points]
            unique_count = len(set(labels))
            if unique_count != len(TennisCourtKeyPointLabel):
                print(name)
    

    def _concat_files(self, extension: str = 'json') -> _RawAnnotations:
        if extension != 'json':
            raise NotImplemented
        
        raw_annotations = []
        for json_file in sorted(self._root_dir.glob(f'*.{extension}')):
            with open(json_file, 'r') as f:
                data = json.load(f)
                raw_annotations.append(
                    {
                        "filename": json_file.name,
                        "data": data
                    }
                ) 
        return raw_annotations


    def filter_by_image(self, image_name: str) -> AT:
        return self.cleaned_annotations.get(image_name)

    @staticmethod
    def display_on_img(
        image: ArrayLike,
        image_name: str,
        collection: 'TennisCourtAnnotationCollection',
        label: str | None = None,
    ) -> None:
        if not collection.cleaned_annotations:
            raise ValueError("Brak cleaned_annotations w kolekcji")
        
        ann = collection.cleaned_annotations.get(image_name)
        if ann is None:
            raise KeyError(f"Brak adnotacji dla obrazu: {image_name!r}")
        
        if label is not None and label not in TennisCourtKeyPointLabel.names():
            raise ValueError(f"Nieprawidłowa etykieta: {label}:")

        arr = np.asarray(image)
        h, w = arr.shape[0], arr.shape[1]

        kps = ann.key_points
        if label is not None:
            lbl = str(label)
            kps = [kp for kp in kps if str(kp.label) == lbl]

        plt.imshow(arr)
        for kp in kps:
            p = transform_keypoint_annotation(arr, [[kp.coordinates.x, kp.coordinates.y]])
            plt.scatter(*p.as_int, label=str(kp.label), s=5)
        plt.show()


    def save(self, file_path: Path | str) -> None:
        if not self.cleaned_annotations:
            print('No data to be saved')
            return

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data_to_save = [item.model_dump() for item in self.cleaned_annotations.values()]

        with open(file_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)

        print('Data saved successfully')


    @staticmethod
    def _build_image_name(item: dict) -> str:
        image_path = item.get("data", {}).get("image", "")
        name = Path(image_path).name if image_path else item.get("file_upload", "")

        if "-" in name:
            return '-'.join(name.split("-", 1)[1:])
        return name
from pathlib import Path

import cv2
from tqdm import tqdm
import tyro

from tennis_court_detection.schemas.testing import TestType
from tennis_court_detection.utils.annotations import TennisCourtAnnotationCollection
from tennis_court_detection.utils.testing import (
    put_text_boxes_overlap, 
    build_output_dir
)

def run(
    test_type: TestType = TestType.ANNOTATIONS,
    images_dir: Path | str = 'data/pics',
    raw_annotations_dir: Path | str = 'data/label-studio-annotations',
    output_dir: Path | str = 'results'
):
    annotations_obj = TennisCourtAnnotationCollection.from_raw_dir(raw_annotations_dir)
    clean_annotations = annotations_obj.cleaned_annotations

    invalid_names = annotations_obj.validate()

    proj_cwd = Path.cwd()
    images_dir = proj_cwd / images_dir
    test_out_dir = build_output_dir(proj_cwd / output_dir, test_type)

    invalid_file = test_out_dir / 'invalid_annotations.txt'
    invalid_file.write_text('\n'.join(invalid_names), encoding='utf-8')

    for pic_name, annotation in tqdm(sorted(clean_annotations.items())):
        img_path = images_dir / pic_name
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if image is None:
            print(f'Image not found {img_path}')
            continue

        h, w = image.shape[:2]

        occupied_boxes = []

        filename_font = cv2.FONT_HERSHEY_SIMPLEX
        filename_font_scale = 0.8
        filename_thickness = 2
        filename_padding = 8

        (filename_w, filename_h), filename_baseline = cv2.getTextSize(
            pic_name,
            filename_font,
            filename_font_scale,
            filename_thickness
        )

        filename_x = 15
        filename_y = 15

        filename_box = (
            filename_x,
            filename_y,
            filename_x + filename_w + 2 * filename_padding,
            filename_y + filename_h + filename_baseline + 2 * filename_padding
        )

        cv2.rectangle(
            image,
            (filename_box[0], filename_box[1]),
            (filename_box[2], filename_box[3]),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            image,
            pic_name,
            (
                filename_x + filename_padding,
                filename_y + filename_h + filename_padding
            ),
            filename_font,
            filename_font_scale,
            (255, 255, 255) if pic_name not in invalid_names else (0, 0, 255),
            filename_thickness,
            cv2.LINE_AA
        )

        occupied_boxes.append(filename_box)

        for kp in annotation.key_points:
            x = int(kp.coordinates.x / 100 * w)
            y = int(kp.coordinates.y / 100 * h)

            cv2.circle(
                image,
                (x, y),
                2,
                (255, 255, 0),
                -1
            )

            text = (
                kp.label.value
                if hasattr(kp.label, 'value')
                else str(kp.label)
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(
                text,
                font,
                font_scale,
                thickness
            )

            offsets = [
                (5, -5),           
                (5, text_h + 5),      
                (-text_w - 5, -5),      
                (-text_w - 5, text_h + 5),
                (5, -text_h - 10),
                (5, 2 * text_h + 10),
                (-text_w - 5, -text_h - 10),
                (-text_w - 5, 2 * text_h + 10),
            ]

            text_position = None
            selected_box = None

            for dx, dy in offsets:
                tx = x + dx
                ty = y + dy

                box = (
                    tx,
                    ty - text_h,
                    tx + text_w,
                    ty + baseline
                )

                if (
                    box[0] < 0
                    or box[1] < 0
                    or box[2] >= w
                    or box[3] >= h
                ):
                    continue

                if any(put_text_boxes_overlap(box, other_box) for other_box in occupied_boxes):
                    continue

                text_position = (tx, ty)
                selected_box = box
                break

            if text_position is None:
                tx = x + 5
                ty = y - 5

                text_position = (tx, ty)

                selected_box = (
                    tx,
                    ty - text_h,
                    tx + text_w,
                    ty + baseline
                )

            occupied_boxes.append(selected_box)

            cv2.putText(
                image,
                text,
                text_position,
                font,
                font_scale,
                (255, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        is_invalid = pic_name in invalid_names
        pic_name, pic_extension = pic_name.rsplit('.', 1)
        pic_name_save = (
            f'{pic_name}_invalid.{pic_extension}'
            if is_invalid
            else f'{pic_name}.{pic_extension}'
        )
        output_path = test_out_dir / pic_name_save

        cv2.imwrite(str(output_path), image)


if __name__ == '__main__':
    tyro.cli(run)
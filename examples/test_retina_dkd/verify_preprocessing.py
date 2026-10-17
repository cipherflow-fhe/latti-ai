"""Compare browser-generated Retina inputs against the supplied Python script."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import openpyxl
from PIL import Image

from prepare_dataset import load_module


def prepare_fixtures(root):
    """Cover non-square down/upscaling, source thresholding, and XLSX cell types."""
    for name in ('images', 'disk', 'lesion'):
        (root / name).mkdir(parents=True, exist_ok=True)
    cases = [
        ('F0001-down.png', 777, 631),
        ('F0002-up.png', 19, 13),
        ('N12345-same.png', 512, 512),
        ('07-nA1234-extra_view.png', 31, 17),
    ]
    for index, (name, width, height) in enumerate(cases):
        y, x = np.indices((height, width))
        rgb = np.stack(
            ((3 * x + 11 * y + 37) % 256, (17 * x + 7 * y + 91) % 256, (13 * x + 19 * y + 5) % 256), axis=2
        ).astype(np.uint8)
        Image.fromarray(rgb).save(root / 'images' / name)
        disk = np.array([0, 79, 80, 81, 127, 128, 255], dtype=np.uint8)[(x // 3 + y // 5) % 7]
        lesion = np.roll(disk, 3, axis=1)
        # A colored mask also checks thresholding independently per source channel.
        if index == 1:
            lesion = np.stack((lesion, np.roll(lesion, 1, axis=0), disk), axis=2)
        for folder, pixels in (('disk', disk), ('lesion', lesion)):
            Image.fromarray(pixels).save(root / folder / f'{Path(name).stem}_{folder}.jpg', quality=100, subsampling=0)
    book = openpyxl.Workbook()
    book.active.append(['index', 'patient_id', 'group', 'press', 'course', 'mashing', 'protein', 'eGFR'])
    for row in [
        (1, 'F0001', 'DN', 120, None, None, 140, 90),
        (2, 'F0002', 'NDRD', 202, 564, 13.8, 207, 132),
        (3, 12345, 'DN', 101, 282, 6.9, 103.5, 66),
        (4, 'A1234', 'NDRD', 0, 0, 0, 0, 0),
    ]:
        book.active.append(row)
    path = root / 'risk_factor_5.xlsx'
    book.save(path)

    # Real subset workbooks use inline strings. Exercise shared strings and a
    # non-default first-sheet ZIP path here without including other patients.
    with ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    namespace = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    sheet = ET.fromstring(parts.pop('xl/worksheets/sheet1.xml'))
    shared = ET.Element(f'{{{namespace}}}sst')
    for cell in sheet.iter(f'{{{namespace}}}c'):
        if cell.get('t') != 'inlineStr':
            continue
        inline = cell.find(f'{{{namespace}}}is')
        value = ''.join(node.text or '' for node in inline.iter(f'{{{namespace}}}t'))
        cell.remove(inline)
        cell.set('t', 's')
        ET.SubElement(cell, f'{{{namespace}}}v').text = str(len(shared))
        ET.SubElement(ET.SubElement(shared, f'{{{namespace}}}si'), f'{{{namespace}}}t').text = value
    parts['xl/worksheets/clinical.xml'] = ET.tostring(sheet, encoding='utf-8')
    parts['xl/sharedStrings.xml'] = ET.tostring(shared, encoding='utf-8')
    relations = ET.fromstring(parts['xl/_rels/workbook.xml.rels'])
    for relation in relations:
        if relation.get('Target', '').endswith('sheet1.xml'):
            relation.set('Target', '/xl/worksheets/clinical.xml')
    ET.SubElement(
        relations,
        '{http://schemas.openxmlformats.org/package/2006/relationships}Relationship',
        {
            'Id': 'rIdShared',
            'Type': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings',
            'Target': 'sharedStrings.xml',
        },
    )
    parts['xl/_rels/workbook.xml.rels'] = ET.tostring(relations, encoding='utf-8')
    types = ET.fromstring(parts['[Content_Types].xml'])
    for entry in types:
        if entry.get('PartName') == '/xl/worksheets/sheet1.xml':
            entry.set('PartName', '/xl/worksheets/clinical.xml')
    ET.SubElement(
        types,
        '{http://schemas.openxmlformats.org/package/2006/content-types}Override',
        {
            'PartName': '/xl/sharedStrings.xml',
            'ContentType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml',
        },
    )
    parts['[Content_Types].xml'] = ET.tostring(types, encoding='utf-8')
    with ZipFile(path, 'w', ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    print(f'Prepared {len(cases)} additional fixtures: {root}')


def compare(reference, dataset, output, normalize):
    ids, non_inv, invasive = reference.read_xlsx_5(str(dataset / 'risk_factor_5.xlsx'))
    results = []
    for image in sorted((dataset / 'images').iterdir()):
        expected = reference.preprocess_image_with_seg(
            str(image),
            str(dataset / 'disk' / f'{image.stem}_disk.jpg'),
            str(dataset / 'lesion' / f'{image.stem}_lesion.jpg'),
            512,
            normalize,
        )
        actual = np.loadtxt(output / 'input_0' / f'{image.stem}.csv', delimiter=',')
        assert actual.shape == (5, 512 * 512), (image.name, actual.shape)
        np.testing.assert_array_equal(actual, expected, err_msg=image.name)
        index = ids[reference.get_id(image.name)]
        errors = []
        for feature, values in (('input_1', non_inv[index]), ('input_2', invasive[index])):
            generated = np.loadtxt(output / feature / f'{image.stem}.csv', delimiter=',')
            assert generated.shape == (len(values),)
            np.testing.assert_allclose(generated, values, atol=1e-15, rtol=0, err_msg=image.name)
            errors.append(float(np.max(np.abs(generated - values))))
        results.append(
            {
                'filename': image.name,
                'shape': list(actual.shape),
                'image_max_abs_error_per_channel': np.max(np.abs(actual - expected), axis=1).tolist(),
                'clinical_max_abs_error': max(errors),
            }
        )
        print(f'Python parity: {image.name}, normalize={normalize}, exact image match', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--client-dir', type=Path, default=Path('/LattiAI-Task-Resources/retina_degree8_02/task/client')
    )
    parser.add_argument('--demo-dir', type=Path)
    parser.add_argument('--prepare-fixtures', action='store_true')
    args = parser.parse_args()
    demo = args.demo_dir or args.client_dir / 'demo_10'
    if args.prepare_fixtures:
        prepare_fixtures(demo / 'validation/fixtures')
        return
    reference = load_module('retina_reference', args.client_dir / 'gen_csv/gen_client_inputs_seg.py')
    actual = compare(reference, demo / 'dataset', demo / 'csv', True)
    with (demo / 'selected_samples.csv').open(newline='', encoding='utf-8') as file:
        selected = list(csv.DictReader(file))
    with (demo / 'dataset/label.csv').open(newline='', encoding='utf-8') as file:
        labels = {row['filename']: int(row['label']) for row in csv.DictReader(file)}
    assert len(selected) == len(labels) == len(actual) == 10
    assert len({row['patient_id'] for row in selected}) == 10
    assert list(labels.values()).count(0) == list(labels.values()).count(1) == 5
    manifest = json.loads((demo / 'manifest.json').read_text())
    ids, non_inv, invasive = reference.read_xlsx_5(str(demo / 'dataset/risk_factor_5.xlsx'))
    for row in selected:
        source, copied = Path(row['image_path']), demo / 'dataset/images' / row['filename']
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest()
            == hashlib.sha256(copied.read_bytes()).hexdigest()
            == manifest['original_images_sha256'][copied.name]
        )
        assert labels[copied.name] == int(row['ground_truth'])
        expected = [float(row[key]) for key in ('non_inv_0', 'non_inv_1', 'inv_0', 'inv_1', 'inv_2')]
        index = ids[row['patient_id']]
        np.testing.assert_allclose(non_inv[index] + invasive[index], expected, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(
            np.loadtxt(demo / 'reference_logits' / f'{copied.stem}.csv', delimiter=','),
            [float(row['raw_output_0']), float(row['raw_output_1'])],
        )
        for kind in ('disk', 'lesion'):
            with Image.open(demo / 'dataset' / kind / f'{copied.stem}_{kind}.jpg') as mask:
                assert mask.size == Image.open(copied).size and mask.mode == 'L'
                mask.verify()
    with (demo / 'reports/predictions.csv').open(newline='', encoding='utf-8') as file:
        predictions = list(csv.DictReader(file))
    by_name = {row['filename']: row for row in selected}
    assert len(predictions) == 10
    for row in predictions:
        source = by_name[row['filename']]
        expected = int(np.argmax([float(source['raw_output_0']), float(source['raw_output_1'])]))
        assert int(row['prediction']) == expected == int(source['predicted'])
        assert int(row['label']) == labels[row['filename']]
        assert int(row['correct']) == int(expected == labels[row['filename']])
    assert sum(int(row['correct']) for row in predictions) == 9
    accuracy = json.loads((demo / 'reports/accuracy.json').read_text())
    assert (accuracy['total'], accuracy['correct'], accuracy['accuracy']) == (10, 9, 0.9)
    fixtures = {}
    for folder, normalize in (('normalized', True), ('unnormalized', False)):
        if (demo / 'validation' / folder).is_dir():
            fixtures[folder] = compare(reference, demo / 'validation/fixtures', demo / 'validation' / folder, normalize)
    report = {
        'status': 'passed',
        'reference_script': str(args.client_dir / 'gen_csv/gen_client_inputs_seg.py'),
        'original_images_unchanged': True,
        'actual_samples': actual,
        'fixtures': fixtures,
        'reference_total': 10,
        'reference_correct': 9,
        'reference_accuracy': 0.9,
        'image_numeric_values_checked': (len(actual) + sum(map(len, fixtures.values()))) * 5 * 512 * 512,
        'image_max_abs_error': 0.0,
        'clinical_tolerance': 1e-15,
    }
    (demo / 'validation/python.json').write_text(json.dumps(report, indent=2) + '\n')
    print(
        f'Verified {len(actual)} real samples and {sum(map(len, fixtures.values()))} fixture outputs; 9/10 historical predictions correct'
    )


if __name__ == '__main__':
    main()

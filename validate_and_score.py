#!/usr/bin/env python3

import os
import sys
import csv
import json
import argparse

# currently hardcoding this for ObjectNet ids
range_min = -1   # oid starts at 0, treat -1 as a valid noop value
range_max = 312

rval = { 'prediction_file_status': 'INVALID',
         'prediction_file_errors': [] }

def err_exit(err_msg):
    rval['prediction_file_errors'].append(err_msg)
    print(json.dumps(rval, indent=2, sort_keys=True))
    sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answers', '-a', required=True, help='ground truth/answer file')
    parser.add_argument('--filename', '-f', required=True, help='users result file')
    parser.add_argument('--range_check', '-r', action='store_true', help='reject entries that have out-of-range label indices')

    try:
        args = parser.parse_args()
    except:
        err_exit('Failed to parse command line')

    try:
        with open(args.answers) as f:
            answers = json.load(f)
        total = len(answers)
    except:
        err_exit('ObjectNet answer file not available')
    
    try:
        f = open(args.filename)
    except:
        err_exit('Unable to open results file: {}'.format(args.filename))

    try:
        reader = csv.reader(f)
    except:
        err_exit('Unable to open results file as csv file: {}'.format(args.filename))

    try:
        linecnt = 0
        num_correct = 0
        num5_correct = 0
        found = set()
        for row in reader:
            try:
                filename = os.path.split(row[0])[1] # remove dir if necessary
            except:
                err_exit('Failure to convert first csv column to string')
            if filename not in answers:
                err_exit('Image name in first csv column not found in answer set')
            if filename in found:
                err_exit('Duplicate image name in result csv..rejecting')
            found.add(filename)

            correct = answers[filename]
            try:
                pred = []
                for x in row[1:10:2]:   # intervleaved, first 5
                    if x == '':   # allow empty predictions
                        pred.append(-1)
                    else:
                        pred.append(int(x))
            except Exception as e:
                # not really happy about swallowing exception, how to dump it w/o the submitter seeing it??
                # print(e, file=sys.stderr)
                err_exit('Failure to convert predictions to integer indices')
            if args.range_check:
                for p in pred:
                    if p < range_min or p > range_max:
                        err_exit('Prediction index <{}> out of range [{}, {}]'.format(p, range_min, range_max))
            if pred[0] == correct:
                num_correct += 1
            if correct in pred:
                num5_correct += 1
            linecnt += 1
    except Exception as e:
        # not really happy about swallowing exception, how to dump it w/o the submitter seeing it??
        # print(e, file=sys.stderr)
        err_exit('Caught exception while parsing csv file: {}'.format(args.filename))

    results = { 'accuracy': 100.0 * num_correct / total,
                'top5_accuracy': 100.0 * num5_correct / total,
                'images_scored': linecnt,
                'total_images': total }
    rval.update(results)
    rval['prediction_file_status'] = 'VALIDATED'
    print(json.dumps(rval, indent=2, sort_keys=True))
    sys.exit(0)
